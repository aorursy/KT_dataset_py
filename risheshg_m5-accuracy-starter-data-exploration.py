# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('bmh')

import seaborn as sns

from sklearn.preprocessing import StandardScaler,MinMaxScaler

min_scaler = MinMaxScaler()

scaler = StandardScaler()

import warnings

warnings.filterwarnings("ignore")

!pip install jovian
import jovian
INPUT_DIR = '../input/m5-forecasting-accuracy'

clndr = pd.read_csv(f'{INPUT_DIR}/calendar.csv')

df_val = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')

submsn = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')

prc = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')

df = pd.read_csv(f'{INPUT_DIR}/sales_train_evaluation.csv')
def sizeof_fmt(num, suffix='B'):

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2 

    print ('Initial Usage = {:5.2f} Mb'.format(start_mem))

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                       df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return None
#reduce_mem_usage(df)

reduce_mem_usage(clndr)

reduce_mem_usage(prc)
clndr['date'] = pd.to_datetime(clndr.date)

clndr['days'] = clndr['date'].dt.day
cln = clndr[:1941]
dates = cln['date']
import time
#da = clndr[0:1913]

da = clndr.copy()
da['event_name_1'] = da['event_name_1'].apply(lambda x: np.where(pd.isnull(x),0,1))

da['event_type_1'] = da['event_type_1'].apply(lambda x: np.where(pd.isnull(x),0,1))

da['event_name_2'] = da['event_name_2'].apply(lambda x: np.where(pd.isnull(x),0,1))

da['event_type_2'] = da['event_type_2'].apply(lambda x: np.where(pd.isnull(x),0,1))
da = da.iloc[:,:14]
da['date'] = pd.to_datetime(da.date)

da.set_index('date',inplace=True)

da.drop(['wm_yr_wk','weekday','d'],axis = 1,inplace=True)
l1 = ['wday','month','year']

for col in l1:

    da[col] = da[col].astype('category')

da = pd.get_dummies(da,columns=l1)

def SMA(data):

    return data.mean()  



def std(data):

    return data.std()



def EMA(data):

    dats = data.astype(float)

    data['EMA'] = dats.ewm(span = 20).mean()

    return data['EMA']
cols = [i for i in df.columns if 'd_' in i ]
date_tr = clndr['date']
def get_sale(item):

    tmp = df.set_index('id').loc[item,cols]

    tmp = tmp.reset_index().drop('index',axis = 1).rename(columns = {0:item})

    return pd.merge(date_tr,tmp,left_index = True,right_index=True).set_index('date')

def dframe(item):

    tmp = da.copy()

    item_sale = get_sale(item)

    

    #tmp.replace('nan', np.nan).fillna(0)

    tmp = pd.merge(tmp,item_sale, left_index=True, right_index=True, how = 'left')

    tmp.rename(columns = {item:'item'},inplace =True)

    

    for i in (1,7,14,28,365):

        tmp['lag_'+str(i)] = tmp['item'].transform(lambda x: x.shift(i))

    

    

    for i in [7,14,28,60,180,365]:

        tmp['rolling_mean_'+str(i)] = tmp['item'].transform(lambda x: x.shift(28).rolling(i).mean())

        tmp['rolling_std_'+str(i)]  = tmp['item'].transform(lambda x: x.shift(28).rolling(i).std())

    

    

    tmp = tmp.replace('nan', np.nan).fillna(0)

    

    return tmp.to_numpy()

    

    

    
from torch.utils.data import Dataset,DataLoader
scaler = StandardScaler()

y_scaler = MinMaxScaler()
'''dx = dframe(item)

xdata = dx.copy()'''
'''trains = xdata[:1500]

tests = xdata[1500:1913]

val = xdata[1885:1941]

evl = xdata[1913:]'''
def sliding_windows_mutli_features(data, seq_length):

    x = []

    y = []

    data_x = scaler.fit_transform(data)

    data_y = data[:,32].reshape(-1,1)



    for i in range((data.shape[0])-seq_length-seq_length+1):

        #print (data.shape[0])

        #print (len(data)-seq_length-1)

        #print (i,(i+seq_length))

        _x = data_x[i:(i+seq_length),:] ## 16 columns for features  

        _y = data_y[i+seq_length:i+seq_length+seq_length] ## column 0 contains the labbel

        #print ('x - ',_x)

        #print ('y - ',_y)

        x.append(_x)

        y.append(_y)



    return np.array(x),np.array(y).reshape(-1,28)
'''train_x,train_y = sliding_windows_mutli_features(trains, seq_length=28)

test_x,test_y = sliding_windows_mutli_features(tests, seq_length=28)

val_x,val_y = sliding_windows_mutli_features(val, seq_length=28)

evl_x,evl_y = sliding_windows_mutli_features(evl, seq_length=28)



print(train_x.shape)

print (train_y.shape)

print (test_x.shape)

print (test_y.shape)

print (val_x.shape)

print (val_y.shape)

print (evl_x.shape)

print (evl_y.shape)



train_set = FeatureDataset(train_x,train_y)

test_set = FeatureDataset(test_x,test_y)

val_set = FeatureDataset(val_x,val_y)

evl_set = FeatureDataset(evl_x,evl_y)



train_loader = DataLoader(dataset = train_set,

                         batch_size = 10

                         )

test_loader = DataLoader(dataset = test_set,

                         batch_size = 10

                         )

val_loader = DataLoader(dataset = val_set,

                         batch_size = 1

                         )

evl_loader = DataLoader(dataset = evl_set,

                         batch_size = 1

                         )'''
class FeatureDataset(Dataset):

    def __init__(self,feature,target):

        self.feature = feature

        self.target = target

    

    def __len__(self):

        return len(self.feature)

    

    def __getitem__(self,idx):

        item = self.feature.reshape(self.feature.shape[0],self.feature.shape[2],self.feature.shape[1])[idx]

        label = self.target[idx]

    

        

        return item,label
def pre_process(xdata,split,seq_length):

    trains = xdata[:split]

    tests = xdata[split:1913]

    val = xdata[1885:1941]

    evl = xdata[1913:]

    

    

    train_x,train_y = sliding_windows_mutli_features(trains, seq_length)

    test_x,test_y = sliding_windows_mutli_features(tests, seq_length)

    val_x,val_y = sliding_windows_mutli_features(val, seq_length=28)

    evl_x,evl_y = sliding_windows_mutli_features(evl, seq_length=28)

    

    train_set = FeatureDataset(train_x,train_y)

    test_set = FeatureDataset(test_x,test_y)

    val_set = FeatureDataset(val_x,val_y)

    evl_set = FeatureDataset(evl_x,evl_y)

    

    train_loader = DataLoader(dataset = train_set,

                         batch_size = 500

                         )

    test_loader = DataLoader(dataset = test_set,

                         batch_size = 300

                         )

    val_loader = DataLoader(dataset = val_set,

                         batch_size = 1

                         )

    evl_loader = DataLoader(dataset = evl_set,

                         batch_size = 1

                         )

    

    return train_loader,test_loader,val_loader,evl_loader
'''pre_process(xdata,1500,28)'''
import torch

import torch.nn as nn

import torch.nn.functional as F

from tqdm import tqdm_notebook as tqdm

from torch.nn.utils import weight_norm

import gc

import collections

import random
class CNN_Forecast(nn.Module):

    def __init__(self,c_in,c_bw,c_out,ks=3,d=2,s=1):

        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(c_in,c_bw,kernel_size=ks,dilation = d,padding = int((d*(ks-1))/2),stride = s))

        self.relu = nn.ReLU(inplace = True)

        self.conv2 = weight_norm(nn.Conv1d(c_bw,c_out,kernel_size=ks,dilation = d,padding = int((d*(ks-1))/2),stride = s))

        self.dp = nn.Dropout(0.2)

        self.shortcut = lambda x: x

        if c_in != c_out:

            self.shortcut = nn.Conv1d(c_in, c_out,kernel_size = 1,stride=1)

        

        y = torch.randn(c_in,28).view(-1,c_in,28)

        self.to_linear = None

        self.convs(y)

        

        self.fc1 = nn.Linear(self.to_linear,512)

        self.fc2 = nn.Linear(512,256)

        self.fc3 = nn.Linear(256,28)

        

    

    def convs(self,x):

        r = self.shortcut(x)

        x = self.relu(self.conv1(x))

        x = self.dp(x)

        x = F.leaky_relu(self.conv2(x),0.1)

        x = self.dp(x)

        #print ('r - ',r.shape)

        #print ('x - ',x.shape)

        r = r.view(x.shape)

        

        if self.to_linear is None:

            self.to_linear = x[0].shape[0]*x[0].shape[1]

        return x.add_(r)

        

    def forward(self,x):

        x = self.convs(x)

        x = x.view(-1,self.to_linear)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        

        return x



model = CNN_Forecast(50,64,32)

print (model)
def conv1d(c_in,c_out,stride= 1,dilation = 1,ks = 3):

    pad = int((dilation*(ks-1))/2)

    return nn.Conv1d(c_in,c_out,kernel_size = ks, stride = stride, dilation = dilation, padding = pad)



def reg(max_ks):

    return nn.Sequential(nn.ReLU(inplace = True),

                         nn.Dropout(0.2),

                         nn.MaxPool1d(max_ks))





def size(s_len,ks,d,c_out,pool,max_ks=1,s=1):

    pad = pad = int((d*(ks-1))/2)

    l_out = (s_len+2*pad-d*(ks-1)-1)/s + 1

    if not pool:

        size = c_out*l_out

    else:

        l_out = (l_out-(max_ks-1)-1)/max_ks + 1

        size = c_out*l_out

    

    return int(size)



    

class ConvNet(nn.Module):

    def __init__(self,n_start,c_out,ks,d,max_ks):

        super().__init__()

        self.conv0 = conv1d(n_start,c_out[0],ks = ks[0],dilation = d[0])

        self.conv1 = conv1d(n_start,c_out[1],ks = ks[1],dilation = d[1])

        self.conv2 = conv1d(n_start,c_out[2],ks = ks[2],dilation = d[2])

        

        self.reg0 = reg(max_ks)

        self.reg1 = reg(max_ks)

        self.reg2 = reg(max_ks)

        

        self.size0 = size(28,ks[0],d[0],c_out[0],max_ks = max_ks,pool = True)

        self.size1 = size(28,ks[1],d[1],c_out[1],max_ks = max_ks,pool = True)

        self.size2 = size(28,ks[2],d[2],c_out[2],max_ks = max_ks,pool = True)

        

        self.fc1 = nn.Linear((self.size0+self.size1+self.size2),100)

        self.fc2 = nn.Linear(100,1)



    def forward(self,x):

        x0 = self.reg0(self.conv0(x))

        x1 = self.reg1(self.conv1(x))

        x2 = self.reg2(self.conv2(x))

        

        x0 = x0.view(-1,self.size0)

        x1 = x1.view(-1,self.size1)

        x2 = x2.view(-1,self.size2)

        

        x = torch.cat([x0,x1,x2],1)

        

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        

        return x

'''device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ConvNet(50,c_out = [50,64,78],ks = [5,7,11],d = [2,2,2],max_ks = 2).double().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)

criterion = nn.MSELoss()'''
'''train_losses = []

valid_losses = []

def Train():

    

    running_loss = .0

    

    model.train()

    

    for idx, (inputs,labels) in enumerate(train_loader):

        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(inputs)

        loss = criterion(preds,labels)

        loss.backward()

        scheduler.step(loss)

        optimizer.step()

        running_loss += loss

        

    train_loss = running_loss/len(train_loader)

    train_losses.append(train_loss.cpu().detach().numpy())

    

    print(f'train_loss {train_loss}')

    

def Valid():

    running_loss = .0

    

    model.eval()

    

    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            preds = model(inputs)

            loss = criterion(preds,labels)

            scheduler.step(loss)

            running_loss += loss

            

        valid_loss = running_loss/len(test_loader)

        valid_losses.append(valid_loss.cpu().detach().numpy())

        print(f'valid_loss {valid_loss}')

        

epochs = 50

for epoch in range(epochs):

    print('epochs {}/{}'.format(epoch+1,epochs))

    Train()

    Valid()

    gc.collect()'''
'''import matplotlib.pyplot as plt

plt.plot(train_losses,label='train_loss')

plt.plot(valid_losses,label='valid_loss')

plt.title('MSE Loss')

plt.xlim(0,epochs)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)'''
'''prediction = [0]*385

running_loss = 0

model.eval()

with torch.no_grad():

        for idx, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            preds = model(inputs)

            prediction[idx] = preds.cpu().detach().numpy()

            loss = criterion(preds,labels)

            running_loss += loss

        test_loss = running_loss/len(test_loader)

        print(f'test_loss {test_loss}')'''
#res = y_scaler.inverse_transform(prediction)

'''res = pd.DataFrame(prediction)

res.index = test_ydates

res.rename({0:'item'},axis =1,inplace=True)

res.reset_index(inplace=True)



#res1 = y_scaler.inverse_transform(prediction)

res1 = pd.DataFrame(prediction)

res1.index = test_ydates

res1.rename({0:'item'},axis =1,inplace=True)

res1.reset_index(inplace=True)



y_tests = pd.DataFrame(test_y)

y_tests.index = test_ydates

y_tests.rename(columns = {0:'item'},inplace = True)



actual = y_tests.reset_index()'''
'''prediction = [0]*28

running_loss = 0

model.eval()

with torch.no_grad():

        for idx, (inputs, labels) in enumerate(val_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            preds = model(inputs)

            prediction[idx] = preds.cpu().detach().numpy()

            loss = criterion(preds,labels)

            running_loss += loss

        test_loss = running_loss/len(val_loader)

        print(f'test_loss {test_loss}')'''
it = ['HOBBIES_1_002_CA_1_validation','HOBBIES_1_004_CA_1_validation','HOBBIES_1_003_CA_1_validation','HOBBIES_1_001_CA_1_validation']
def Trainings1(product):

    running_loss = .0

    model.train()

    

    for idx, (inputs,labels) in enumerate(train_loader):

        inputs = inputs.to(device)

        labels = labels.to(device,non_blocking = True)

        optimizer.zero_grad()

        preds = model(inputs)

        loss = criterion(preds,labels)

        loss.backward()

        #scheduler.step(loss)

        optimizer.step()

        running_loss += loss

        

        #print (f"epoch {epoch}, datapoints {idx*10}")

    

    train_loss = running_loss/len(train_loader)

    train_losses[str(product)].append(train_loss.cpu().detach().numpy().item())

        

        

def Validations1(product):

    running_loss = .0

    model.eval()

    

    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)

            labels = labels.to(device,non_blocking = True)

            optimizer.zero_grad()

            preds = model(inputs)

            loss = criterion(preds,labels)

            #scheduler.step(loss)

            running_loss += loss

            #print (f"epoch {epoch}, datapoints {idx*10}")

            

        valid_loss = running_loss/len(test_loader)

        valid_losses[str(product)].append(valid_loss.cpu().detach().numpy().item())

            

def Predicts1(product):

    with torch.no_grad():

            for idx, (inputs,_) in enumerate(val_loader):

                inputs = inputs.to(device)

                optimizer.zero_grad()

                preds = model(inputs)

                pred_dict.update({product:preds.cpu().detach().numpy().reshape(28,).tolist()})



            

            
train_cv_loss = pd.DataFrame()

valid_cv_loss = pd.DataFrame()
'''start1 = time.time()

#start = time.time()

seq_length = 28

spilt = 1500



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()

epochs = 20



train_losses = collections.defaultdict(list)

valid_losses = collections.defaultdict(list)

prediction = [0]*28

pred_dict = {}

#print ('Time Taken for Initialisation = {}s'.format((time.time() - start))*100)



first = True



model = CNN_Forecast(50,64,32,ks = 11,d=3).double().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)



for product in tqdm(df_val['id']):

    

    

    #print (product)

    #start = time.time()

    dset = dframe(product)

    #dset = np.load('.'.join([product,'npy']))

    #print ('Time Taken for Creating Dataset = {}s'.format((time.time() - start))*100)

    #print ('-----Dateset Created-----')

    #start = time.time()

    train_loader,test_loader,val_loader,_ = pre_process(dset,split,seq_length)

    #print ('Time Taken for Creating Loaders = {}s'.format((time.time() - start))*100)

    #print ('Loaders Ready !!!')



    #print ('Training')

    #start = time.time()

    for epoch in range(epochs):

        Trainings1(product)

        Validations1(product)

    #print ('Time Taken for Validating Model = {}s'.format((time.time() - start))*100)

    Predicts1(product)

train_losses = pd.DataFrame(train_losses)

valid_losses = pd.DataFrame(valid_losses)

pred_d = pd.DataFrame(pred_dict)

train_cv_loss = pd.concat([train_cv_loss,train_losses],axis = 1)

valid_cv_loss = pd.concat([valid_cv_loss,valid_losses],axis = 1)

train_losses = collections.defaultdict(list)

valid_losses = collections.defaultdict(list)

    

    

end = time.time()

print('Time taken is {}s'.format(end-start1))'''
def Trainings1(product):

    model.train()

    

    for idx, (inputs,labels) in enumerate(train_loader):

        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(inputs)

        loss = criterion(preds,labels)

        loss.backward()

        #scheduler.step(loss)

        optimizer.step()

        

        

def Validations1(product):

    model.eval()

    

    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            preds = model(inputs)

            loss = criterion(preds,labels)

            #scheduler.step(loss)



            

def Predicts1(product):

    with torch.no_grad():

            for idx, (inputs,_) in enumerate(evl_loader):

                inputs = inputs.to(device)

                optimizer.zero_grad()

                preds = model(inputs)

                pred_dict.update({product:preds.cpu().detach().numpy().reshape(28,).tolist()})



            

            
idir_torch1 = '../input/torch-model'

loaded = torch.load(f'{idir_torch1}/cpoints.pth')
idir_torch2 = '../input/torch-model-new'

loaded = torch.load(f'{idir_torch2}/cpoints_new.pth')
'''start1 = time.time()

#start = time.time()

seq_length = 28

split = 1500



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()

epochs = 20



prediction = [0]*28

pred_dict = {}



first = True



model = CNN_Forecast(50,64,32,ks = 11,d=3).double()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)



idir_torch2 = '../input/torch-model-new'

loaded = torch.load(f'{idir_torch2}/cpoints_new.pth')



model.load_state_dict(loaded['model_state'])

optimizer.load_state_dict(loaded['optim_state'])

model.to(device)

for state in optimizer.state.values():

    for k, v in state.items():

         if isinstance(v, torch.Tensor):

            state[k] = v.cuda()

            

for product in tqdm(df['id']):

    

    

    dset = dframe(product)

    #dset = np.load('.'.join([product,'npy']))

    train_loader,test_loader,val_loader,evl_loader = pre_process(dset,split,seq_length)

    

    for epoch in range(epochs):

        Trainings1(product)

        Validations1(product)

   

    Predicts1(product)



pred_d = pd.DataFrame(pred_dict).T

pred_d.to_csv('predictions1.csv')    

    

end = time.time()



print('Time taken is {}s'.format(end-start1))'''
pd.DataFrame(pred_dict).T
checkpoint = {

        "model_state":model.state_dict(),

        "optim_state":optimizer.state_dict()

    }

    

torch.save(checkpoint,'cpoints_new.pth')
pred_d.to_csv('predictions_evl.csv')
pred_d
#64,32 bs = 500

#28,28 bs = 500

#100,200 bs = 500

#64,32 ks = 20 bs = 500

#50,64,32,ks = 11,d=4 bs = 1913

#bs = 200 and 100

#bs = 500 and 300 64,32 11,3 e = 20

#e = 15

#e = 30

#e = 5

#e = 15 t = 8hrs for saved else 11 hrs

#e = 20 t = 13hrs for not saved else 11.25

#e = 20 t = 10hr no scheduler saved model

#e= 20 t = 12hrs with scheduler saved model

#e= 20 t = 10.5hrs with no scheduler non saved model



#ConvNet(50,c_out = [50,64,78],ks = [5,7,11],d = [2,2,2],max_ks = 2)

#CNN_Forecast(50,100,150)

#ConvNet(50,c_out = [50,64,78],ks = [5,7,11],d = [2,3,4],max_ks = 2)

#ConvNet(50,c_out = [50,64,78],ks = [7,11,13],d = [2,5,7],max_ks = 2)

#ConvNet(50,c_out = [50,64,78],ks = [7,11,13],d = [2,5,7],max_ks = 2) without batch normalization

#CNN_Forecast(50,64,32,ks = 11,d=3)
idir_dset = '../input/predictions'

loaded_dset1 = pd.read_csv(f'{idir_dset}/predictions1.csv')



idir_dset2 = '../input/predictions2'

loaded_dset2 = pd.read_csv(f'{idir_dset2}/predictions2.csv')



idir_dset3 = '../input/predictions-evl'

loaded_dset3 = pd.read_csv(f'{idir_dset3}/predictions_evl.csv')
preds = pd.concat([loaded_dset1,loaded_dset2,loaded_dset3])
preds.columns = submsn.columns
preds
submsn
preds.to_csv('submission1.csv')
idir_sb = '../input/m5-first-public-notebook-under-0-50'

submt_dset2 = pd.read_csv(f'{idir_sb}/submission_2.csv')
preds[preds['F25']<0]