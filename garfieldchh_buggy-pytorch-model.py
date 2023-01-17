import torch
from torch.autograd import Variable
from torch import nn,optim
from torchvision import transforms
import torchvision.datasets as dset
import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
! cp /kaggle/input/* ./

# Processing Training data
def proc_df(train, window_size):
    train_y = train.iloc[window_size:]
    
    train_proc = pd.DataFrame(index=train["Date"][window_size: ], \
                          columns=[[ f'day_t-{window_size + 1 - i}'for i in range(1,window_size + 1)] ])
    
    for i in range(len(train_proc)):

        train_proc.iloc[i] = train["Open"][i: i + window_size].values
        
    train_proc = train_proc.values
#     test_proc = test_proc.values
    train_y = train_y["Open"].values 
    train_y = np.reshape(train_y, (-1,1))
    train_proc = np.reshape(train_proc, (train_proc.shape[0], train_proc.shape[1], 1))
    train_proc = train_proc.astype(np.float32)
    train_y = train_y.astype(np.float32)
    return train_proc, train_y

# Pytorch dataset for processed training data
class Stock_DS(torch.utils.data.Dataset):
    def __init__(self,data, label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)
        
    def __getitem__(self, index):
        return (self.data[index], self.label[index])
    def __len__(self):
        return len(self.data)
    
# Prepare test data
def get_test(window_size, sc):
    test_proc = pd.DataFrame(index=full["Date"][-20:],\
                          columns=[[ f'day_t-{window_size + 1 - i}'for i in range(1,window_size + 1)] ])

    for i in range(len(test_proc)):
            test_proc.iloc[i] = full["Open"][ len(full) - 20 + i - window_size : len(full) - 20 + i].values
    test_proc = test_proc.values
    test_proc = sc.transform(test_proc)
    test_proc = np.reshape(test_proc, (-1,window_size,1))
    test_proc = test_proc.astype(np.float32)
    return test_proc
 
# Get X_train y_train for keras model
train = pd.read_csv("Google_Stock_Price_Train.csv")[["Date","Open"]]
test = pd.read_csv("Google_Stock_Price_Test.csv")[["Date","Open"]]
sc = MinMaxScaler()
train["Open"] = sc.fit_transform(train["Open"].values.reshape(-1,1))
X_train, y_train = proc_df(train, window_size=60)

# Get pytorch dataloader for pytorch model
ds = Stock_DS(X_train, y_train)
dl = torch.utils.data.dataloader.DataLoader(ds, batch_size=32, shuffle=False)

# Get X_test for keras model and X_test_pytorch for pytorch model
train_raw = pd.read_csv("Google_Stock_Price_Train.csv")[["Date","Open"]]
test_raw = pd.read_csv("Google_Stock_Price_Test.csv")[["Date","Open"]]
full = pd.concat([train_raw, test_raw],axis=0)
X_test = get_test(60, sc)
X_test_pytorch = torch.from_numpy(X_test).cuda()
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout
regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = "adam", loss = 'mean_squared_error')

regressor.summary()
regressor.fit(X_train,y_train, epochs=100, batch_size = 32)

y_preds_keras = regressor.predict(X_test)
y_preds_unscaled = sc.inverse_transform(y_preds_keras)
plt.plot(test_raw["Open"] )
plt.xlabel("time")
plt.ylabel("price")
plt.plot(y_preds_unscaled )
plt.xlabel("time")
plt.ylabel("price")

def my_training_loop(m, dl, epochs):
    opt = optim.Adam(m.parameters())
    crit = nn.MSELoss()
    
    for epoch in range(epochs):
        accu_loss = 0
        batch_count = 0
        for i, (train_x, train_y) in enumerate(dl):

            x = Variable(train_x.cuda())
            y = train_y.cuda()
            opt.zero_grad()
            preds = m(x)
            loss = crit(preds, y)
            accu_loss += loss.item()
            batch_count += 1
            loss.backward()

            opt.step()
        print(f'Epoch: {epoch}. Loss: {accu_loss/batch_count}')
class MyLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        
        super().__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,\
                          num_layers=layer_dim,bias=True, batch_first = True,dropout=0.2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, x):
        h0 = Variable(torch.zeros((self.layer_dim, x.size(0), self.hidden_dim)).cuda())
        c0 = Variable(torch.zeros((self.layer_dim, x.size(0), self.hidden_dim)).cuda())

        o, h = self.lstm(x, (h0,c0))
        o = self.fc(self.dropout(o[:,-1,:]))
        return o 
m = MyLSTM(1, 50, 4, 1)
m.cuda()
my_training_loop(m, dl, 100)
y_preds_torch = m(X_test_pytorch)
y_preds_torch_unscaled = sc.inverse_transform(y_preds_torch.cpu().data)
plt.plot(test_raw["Open"] )
plt.xlabel("time")
plt.ylabel("price")
plt.plot(y_preds_torch_unscaled)
plt.xlabel("time")
plt.ylabel("price")
