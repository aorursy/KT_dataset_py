import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

print(os.listdir("../input"))
train = pd.read_csv("../input/train_V2.csv")
print(train.shape)
train.head()
#device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#defining utility class
#by defining this, you only have to write "for loop" to load minibatch data
class DataLoader(object):
    def __init__(self, x, y, batch_size=128, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_idx = 0
        self.data_size = x.shape[0]
        if self.shuffle:
            self.reset()
    
    def reset(self):
        self.x, self.y = shuffle(self.x, self.y)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start_idx >= self.data_size:
            if self.shuffle:
                self.reset()
            self.start_idx = 0
            raise StopIteration
    
        batch_x = self.x[self.start_idx:self.start_idx+self.batch_size]
        batch_y = self.y[self.start_idx:self.start_idx+self.batch_size]

        batch_x = torch.tensor(batch_x, dtype=torch.float, device=device)
        batch_y = torch.tensor(batch_y, dtype=torch.float, device=device)

        self.start_idx += self.batch_size

        return (batch_x,batch_y)

#defining MLP model
#generally out_dim is more than 1, but this model only allows 1.
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super(MLP, self).__init__()
        
        assert out_dim==1, 'out_dim must be 1'
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.out_dim)
    
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = x.squeeze(1)
        return x
#data formatting
y = train.winPlacePerc
x = train.drop(['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc'], axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

# pd.DataFrame to np.ndarray
x_train = x_train.values
y_train = y_train.values
x_valid = x_valid.values
y_valid = y_valid.values
assert isinstance(x_train, np.ndarray)
#instantiate model
mlp = MLP(x_train.shape[1], 200, 1).to(device)
optimizer = optim.Adam(mlp.parameters())
train_dataloader = DataLoader(x_train, y_train, batch_size=4000)
valid_dataloader = DataLoader(x_valid, y_valid, batch_size=4000)
#this model learns to minimize MAE
def mae_loss(y_pred, y_true):
    mae = torch.abs(y_true - y_pred).mean()
    return mae
#training phase
epochs = 20
#to plot loss curve after training
valid_losses = []

for epoch in range(epochs):
    start_time = time.time()
    mlp.train()
    num_batch = train_dataloader.data_size // train_dataloader.batch_size + 1
    
    for batch_id, (batch_x, batch_y) in enumerate(train_dataloader):
        
        y_pred = mlp(batch_x)

        loss = mae_loss(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed_time = time.time() - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = elapsed_time - 60 * elapsed_min

        print('\rEpoch:{} Batch:{}/{} Loss:{:.4f} Time:{}m{:.2f}s'.format(epoch + 1, batch_id, 
                                                                          num_batch, loss.item(),
                                                                          elapsed_min, elapsed_sec), end='')
    print()
    mlp.eval()
    valid_loss = 0
    best_loss = np.inf
    num_batch = valid_dataloader.data_size // valid_dataloader.batch_size + 1
    
    for batch_id, (batch_x, batch_y) in enumerate(valid_dataloader):
    
        y_pred = mlp(batch_x)
        loss = mae_loss(y_pred, batch_y)
        valid_loss += loss.item()
    
    valid_loss /= num_batch
    valid_losses.append(valid_loss)
    
    #save model when validation loss is minimum
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(mlp.state_dict(), 'mlp.model')  
    
    print('Valid Loss:{:.4f}'.format(valid_loss))
#plot validation loss curve, this may help to notice overfitting
plt.figure(figsize=(16,5))
plt.ylim(0,max(valid_losses)+0.02)
plt.plot(valid_losses)
print('minimum validation loss is {:.4f}'.format(min(valid_losses)))
#load the best model
mlp.load_state_dict(torch.load('mlp.model'))

test = pd.read_csv('../input/test_V2.csv')
#data formatting
x_test = test.drop(['Id', 'groupId', 'matchId', 'matchType'],axis=1)
x_test = torch.tensor(x_test.values,dtype=torch.float,device=device)

#predict
y_pred = mlp(x_test)
y_pred = y_pred.data.cpu().numpy()

#format to csv file
y_pred = pd.DataFrame(y_pred,columns=['winPlacePerc'])
y_pred['Id'] = test['Id']
y_pred = y_pred[['Id', 'winPlacePerc']]
y_pred.to_csv('submission.csv',index=False)
!head submission.csv