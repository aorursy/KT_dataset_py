# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics
df = pd.read_csv('/kaggle/input/daily-historical-stock-prices-1970-2018/historical_stock_prices.csv',index_col=0)
df.info()
df.head()
## df.shape
COLUMN_NAMES = ['open','close','adj_close','low','high']
df_ge = df.loc['CAT',COLUMN_NAMES]
df_ge.head()
print(df.shape)
print(df_ge.shape)
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def build_timeseries1D(mat, y_col_index, TIME_STEPS):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 =1
    x = np.zeros((dim_0, TIME_STEPS,dim_1))
    y = np.zeros((dim_0,dim_1))
    
    for i in tqdm.notebook.tqdm(range(dim_0)):
        aux = mat[i:TIME_STEPS+i, y_col_index]
        x[i] = np.expand_dims(aux,axis = 1)
        y[i] = mat[TIME_STEPS+i, y_col_index]
        y[i] = np.expand_dims(y[i],axis = 1)
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 256 # number of hidden states
        self.n_layers = 2 # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(n_features, 
                                 self.n_hidden,
                                 self.n_layers, 
                                 batch_first = True, bidirectional = False, dropout = 0.2)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(in_features = self.n_hidden, out_features = 1)
        

    def forward(self, x, h):        
        batch_size, seq_len, _ = x.size()
        
        #Dudas sobre esta parte *****
        #(Darle margen de error en los primeros datos)
        x, self.hidden = self.l_lstm(x,h)
        #x = x[:,-1,:]
        x = x[:,-1]
        x = F.relu(x)
        x = self.l_linear(x)
        
        return x, self.hidden
    
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers,batch_size,self.n_hidden).zero_().to(device),weight.new(self.n_layers,batch_size,self.n_hidden).zero_().to(device))
        return(hidden)

#Se usa esta:
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
X = df_ge.values
X.shape


Xtrain, Xtest = train_test_split(X,train_size=0.9, test_size=0.1, shuffle=False)
Xtr, Xval = train_test_split(Xtrain,train_size=0.9, test_size=0.1, shuffle=False)
print("Shapes")
print("Xtrain:",Xtrain.shape,"\nXtr:", Xtr.shape)
print("Xtest:",Xtest.shape,"\nXval:",Xval.shape)
y_col_index = 2
scal_lab = MinMaxScaler()
scal_lab_fit = scal_lab.fit(Xtr[:,y_col_index].reshape(-1, 1))
scal = MinMaxScaler()
Xtr = scal.fit_transform(Xtr)
Xval = scal.transform(Xval)
Xtest = scal.transform(Xtest)
#Create train, validation, and test sets
#cols = df_ge.columns.values.tolist()
#cols = cols[1:]

X = df_ge.values
#scal = MinMaxScaler()
#X = scal.fit_transform(X)

#df_train, df_test = train_test_split(df_ge, train_size=0.9, test_size=0.1, shuffle=False)
Xtrain, Xtest = train_test_split(X,train_size=0.9, test_size=0.1, shuffle=False)
Xtr, Xval = train_test_split(Xtrain,train_size=0.9, test_size=0.1, shuffle=False)

y_col_index = 2
scal_lab = MinMaxScaler()
scal_lab_fit = scal_lab.fit(Xtr[:,y_col_index].reshape(-1, 1))


scal = MinMaxScaler()
Xtr = scal.fit_transform(Xtr)
Xval = scal.transform(Xval)
Xtest = scal.transform(Xtest)


#del df_ge
#del df_train


#Create time series and batchify

tw = 90
batch_size = 64
Xtr, ytr = build_timeseries1D(Xtr,y_col_index,tw)
Xval, yval = build_timeseries1D(Xval,y_col_index,tw)
Xtest, ytest = build_timeseries1D(Xtest,y_col_index,tw)


train_data = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

val_data = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
plt.plot(ytr)
n_features = 1 # this is number of parallel inputs
n_timesteps = tw # this is number of timesteps
hidden_dim = 256
output_dim = 1
n_layers = 2
# create NN
#model = MV_LSTM(n_features,n_timesteps)

model = LSTMNet(n_features,hidden_dim,output_dim,n_layers)
model.to(device)

criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_episodes = 60
model.train()
for t in range(train_episodes):
    htr = model.init_hidden(batch_size)
    for x,y in train_loader:
        htr = tuple([e.data for e in htr])
        model.zero_grad()
        output, htr = model(x.to(device).float(),htr) 
        loss = criterion(output, y.to(device).float())  
        #Calcular gradiente
        loss.backward() 
        #Mover los datos
        optimizer.step()        
        
    hval = model.init_hidden(batch_size)    
    for xval,yval in val_loader:
        hval = tuple([e.data for e in hval])
        output_val, hval = model(xval.to(device).float(),htr) 
        loss_val = criterion(output_val, yval.to(device).float())  
        
        
        
    print('step : ' , t , ' loss_train: ' , loss.item(), ' loss_val: ', loss_val.item())
#Evaluate the model
model.eval()
Xtest = torch.from_numpy(Xtest)
htest = model.init_hidden(Xtest.shape[0])
out, htest = model(Xtest.to(device).float(), htest)

out = out.cpu().detach().numpy()
out = scal_lab_fit.inverse_transform(out)
ytest = scal_lab_fit.inverse_transform(ytest)
fig = plt.figure()

ax11 = fig.add_subplot(211)
ax11.plot(out)
ax11.plot(ytest,'r')