import pandas as pd

import numpy as np

from scipy.special import erfinv

import matplotlib.pyplot as plt

import torch

from torch.utils.data import *

from torch.optim import *

from fastai.tabular import *

import torch.utils.data as Data

from fastai.basics import *

from fastai.callbacks.hooks import *

from tqdm import tqdm_notebook as tqdm

import gc

import joblib



%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train_test = pd.concat([train, test]).reset_index()
#train_test = train_test.sample(frac=0.5)
train_test.reset_index(inplace=True)
train_test.head()
def preprocess(data):

    usecols = [c for c in data.columns]

    data = data.loc[:,usecols]

    #cat_features = [c for c in data.columns if 'cat' in c]

    #add_df = data[cat_features]

    #data = pd.get_dummies(data, columns=cat_features)

    #data = pd.concat([data, add_df], axis= 1).drop('index', 1)

    data = data.drop(['index', 'level_0'], 1)

    return data
data = preprocess(train_test)
data.head()
def to_gauss(x): return np.sqrt(2)*erfinv(x)  #from scipy



def normalize(data, exclude=None):

    # if not binary, normalize it

    norm_cols = [n for n, c in data.drop(exclude, 1).items() if len(np.unique(c)) > 2]

    n = data.shape[0]

    for col in norm_cols:

        sorted_idx = data[col].sort_values().index.tolist()# list of sorted index

        uniform = np.linspace(start=-0.99, stop=0.99, num=n) # linsapce

        normal = to_gauss(uniform) # apply gauss to linspace

        normalized_col = pd.Series(index=sorted_idx, data=normal) # sorted idx and normalized space

        data[col] = normalized_col # column receives its corresponding rank

    return data
norm_data = normalize(data, exclude=['ID_code', 'target'])
dropcols= ['ID_code', 'target']

#X = np.array(norm_data.drop(dropcols, 1))

save = norm_data.loc[:, dropcols+['var_0']]

X = norm_data.drop(dropcols, 1)
X.shape
save.shape
del norm_data

del train

del test

del train_test

del data
def inputSwapNoise(arr, p):

    ### Takes a numpy array and swaps a row of each 

    ### feature with another value from the same column with probability p

    

    n, m = arr.shape

    idx = range(n)

    swap_n = round(n*p)

    for i in range(m):

        col_vals = np.random.permutation(arr[:, i]) # change the order of the row

        swap_idx = np.random.choice(idx, size= swap_n) # choose row

        arr[swap_idx, i] = np.random.choice(col_vals, size = swap_n) # n*p row and change it 

    return arr
class BatchSwapNoise(nn.Module):

    """Swap Noise module"""



    def __init__(self, p):

        super().__init__()

        self.p = p



    def forward(self, x):

        if self.training:

            mask = torch.rand(x.size()) > (1 - self.p)

            idx = torch.add(torch.arange(x.nelement()),

                            (torch.floor(torch.rand(x.size()) * x.size(0)).type(torch.LongTensor) *

                             (mask.type(torch.LongTensor) * x.size(1))).view(-1))

            idx[idx>=x.nelement()] = idx[idx>=x.nelement()]-x.nelement()

            return x.view(-1)[idx].view(x.size())

        else:

            return x
class ArraysItemList(FloatList):

    def __init__(self, items:Iterator, log:bool=False, **kwargs):

        if isinstance(items, ItemList):

            items = items.items

        super(FloatList,self).__init__(items,**kwargs)

    

    def get(self,i):

        return Tensor(super(FloatList,self).get(i).astype('float32'))
x_il = ArraysItemList(X)

x_ils = x_il.split_by_rand_pct()

lls = x_ils.label_from_lists(x_ils.train, x_ils.valid)

data = lls.databunch(bs=32)
x,y = next(iter(data.train_dl))

x.shape,y.shape
data.train_ds
class Autoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.noise = BatchSwapNoise(0.15)

        self.encoder = nn.Sequential(

            nn.Linear(200, 300),

            nn.Linear(300, 300),

            nn.Linear(300, 300)

        )

        self.decoder = nn.Sequential(

            nn.Linear(300, 300),

            nn.Linear(300, 200)

        )



    def forward(self, xb): 

        encoder = self.encoder(self.noise(xb))

        decoder = self.decoder(encoder)

        return decoder
model = Autoencoder().cuda()
loss_func = F.mse_loss
learn = Learner(data, Autoencoder(), loss_func=loss_func)
learn.lr_find()

learn.recorder.plot(stop_div=False)
learn.fit_one_cycle(10)
learn.recorder.plot_losses(True)
learn.recorder.plot_lr(True)
m = learn.model.eval()

joblib.dump(m, open('m.p', 'wb'))
#x,y = data.train_ds[0]

# have to convert into batch before putting into the model

#xb,_ = data.one_item(x)

#xb = xb.cuda()
def hooked_backward(xb, cat=y):

    with hook_output(m.encoder[0]) as hook_a: 

        with hook_output(m.encoder[1]) as hook_b: 

            with hook_output(m.encoder[2]) as hook_c: 

                with hook_output(m.decoder[0]) as hook_d: 

                    with hook_output(m.decoder[1]) as hook_e: 

                        preds = m(xb)

    return hook_a, hook_b, hook_c, hook_d, hook_e
#hook_a, hook_b, hook_c, hook_d, hook_e = hooked_backward(xb)

#acts_a = hook_a.stored[0].cpu()

#acts_b = hook_b.stored[0].cpu()

#acts_c = hook_c.stored[0].cpu()

#acts_d = hook_d.stored[0].cpu()

#acts_e = hook_e.stored[0].cpu()



#a = np.concatenate((acts_a, acts_b, acts_c, acts_d, acts_e))

#b = np.concatenate((acts_a, acts_b, acts_c, acts_d, acts_e))

#result_array = np.empty((0, 1400))



#np.vstack((result_array, a)).shape
data = None

gc.collect()
X1 = X.iloc[:int(X.shape[0]*0.25), ]
x_il = ArraysItemList(X1)

x_ils = x_il.split_none()

lls = x_ils.label_from_lists(x_ils.train, [])

data = x_ils.databunch(bs=32)
def extract_features(data, learner):

    len_data = len(data.train_ds)

    result = np.empty((len_data, 1400))

    for i in tqdm(range(len_data)):

        x,y = data.train_ds[i]

        xb,_ = data.one_item(x)

        xb = xb.cuda()

        hook_a, hook_b, hook_c, hook_d, hook_e = hooked_backward(xb)

        

        acts_a = hook_a.stored[0].cpu()

        acts_b = hook_b.stored[0].cpu()

        acts_c = hook_c.stored[0].cpu()

        acts_d = hook_d.stored[0].cpu()

        acts_e = hook_e.stored[0].cpu()

        result[i] = np.concatenate((acts_a, acts_b, acts_c, acts_d, acts_e))

        

    return result
result = extract_features(data, learn)

#data = None 

#x_il = None

#x_ils = None 

#lls = None

#gc.collect()
#a = pd.DataFrame(result)

#result = None

#gc.collect()
#a = pd.concat([save.iloc[:int(X.shape[0]*0.25), :2], a], axis=1)
#a.head()
joblib.dump(a, open('a.p', 'wb'))

a = None

gc.collect()
#X2 = X.iloc[int(X.shape[0]*0.25):int(X.shape[0]*0.5), ]

#x_il = ArraysItemList(X2)

#x_ils = x_il.split_none()

#lls = x_ils.label_from_lists(x_ils.train, [])

#data = x_ils.databunch(bs=32)
#result = extract_features(data, learn)

#data = None 

#x_il = None

#x_ils = None 

#lls = None

#gc.collect()



#b = pd.DataFrame(result)

#b = pd.concat([save.iloc[int(X.shape[0]*0.25):int(X.shape[0]*0.5), :2], b], axis=1)
#joblib.dump(b, open('b.p', 'wb'))

#b = None

#gc.collect()
#X3 = X.iloc[int(X.shape[0]*0.5):int(X.shape[0]*0.75), ]

#x_il = ArraysItemList(X3)

#x_ils = x_il.split_none()

#lls = x_ils.label_from_lists(x_ils.train, [])

#data = x_ils.databunch(bs=32)
##result = extract_features(data, learn)

#data = None 

#x_il = None

#x_ils = None 

#lls = None

#gc.collect()

#c = pd.DataFrame(result)

#c = pd.concat([save.iloc[int(X.shape[0]*0.5):int(X.shape[0]*0.75), :2], c], axis=1)
##joblib.dump(c, open('c.p', 'wb'))

#c = None

#gc.collect()
#X4 = X.iloc[int(X.shape[0]*0.75):, ]

#x_il = ArraysItemList(X4)

#x_ils = x_il.split_none()

#lls = x_ils.label_from_lists(x_ils.train, [])

#data = x_ils.databunch(bs=32)
#result = extract_features(data, learn)



#data = None 

#x_il = None

#x_ils = None 

#lls = None

#gc.collect()



#d = pd.DataFrame(result)

#d = pd.concat([save.iloc[int(X.shape[0]*0.75):, :2], c], axis=1)
#joblib.dump(d, open('d.p', 'wb'))

#d = None

#gc.collect()