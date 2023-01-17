import torch as T
import pandas as pd

def split_to_train_test(df, label_column, train_frac=0.8):
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        print('\n{}:\n---------\ntotal:{}\ntrain_df:{}\ntest_df:{}'.format(lbl, len(lbl_df), len(lbl_train_df), len(lbl_test_df)))
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)

    return train_df, test_df

data = pd.read_pickle('../input/mfcc_pandas.pbz2',compression='bz2')
df_train, df_test = split_to_train_test(data, 'target')
df_test.head()
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class Dataset(Dataset):
    """
    Kelas ini adalah kelas yang digunakan untuk menampung dataset.
    """
    def __init__(self, df):
        """
        Inisiasi dataset dengan file yang digunakan
        """        
        self.attrs = df['mfcc'].values
        self.unik = df['target'].unique()
        self.target = df['target'].values
    
    def __len__(self):
        """
        Mengembalikan jumlah dari dataset
        """
        return len(self.target)

    def __getitem__(self, index):
        """
        Untuk mengembalikan satu buah data dari dataset
        index : Index dari data yang akan diambil
        return X : 
        """
        target = T.tensor(self.onehot(self.target[index]))
        attrs = T.tensor(self.attrs[index])
        attrs = attrs.view(attrs.numel())
        return attrs, target
    
    def onehot(self, values):
        vector = np.zeros(len(self.unik))
        for i in range(0, len(self.unik)):
            if self.unik[i] == 'Aaron_Yoo':
                vector[i] = 1            
        return vector
ds_train = Dataset(df_train)
ds_test = Dataset(df_test)
dl_train = DataLoader(ds_train, batch_size=4, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=4, shuffle=True)
import torch.nn as nn

class ModelClass(nn.Module):
    def __init__(self,inp,out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ModelClass,self).__init__()
        self.layer_input = nn.Linear(inp, 30)
        self.layer_hidden = nn.Linear(30,15)
        self.layer_out = nn.Linear(15,out)

        
    def forward(self,x,leng=13):
        """
        Ini adalah fungsi untuk menjalankan ANNnya,
        
        """
        x = T.sigmoid(self.layer_input(x))
        x = T.sigmoid(self.layer_hidden(x))
        x = T.sigmoid(self.layer_out(x))
        
        return x
    
    
n_in = 728
n_out = 100
model = ModelClass(n_in,n_out)
criterion = nn.MultiLabelSoftMarginLoss()
    
optimizer = T.optim.Adam(model.parameters(),1e-4)
epoch = 10
LOSS = []
import tqdm

model = model.train()
for _ in tqdm.tqdm(range(epoch)):
    a=[]
    for i_batch, sample_batched in enumerate(dl_train):
        x, y = sample_batched
        x = x.float()
        y = y.float()
        y_i = model(x)
        
        #hitung loss
        loss = criterion(y_i, y)
        a += [loss.mean()]
        
        #SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    LOSS += [sum(a)/len(a)]
    
print("done")
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(LOSS)
plt.show()
model = model.eval()
cor = tot = 0
rerata = []
for i_batch, sample_batched in enumerate(dl_test):
        x, y = sample_batched
        #print(T.max(y,0))
        x = x.float()
        y = y.float()
        #print(y)
        y_i = model(x)
        _,i = T.max(y,1)
        values,u = T.max(y_i,1)
        for a in range(len(u)):
            #print(i[a].item())
            if i[a] == u[a]:
                cor += 1 
                rerata.append([values[a].item(), i[a].argmax().item()])
                #rerata.append((u[a].item(),i[a]))
        tot += len(u)

print("Akurasi = {} %".format(cor/tot*100))
print(tot)
T.save(model.state_dict(), 'model_state_dict.pth')