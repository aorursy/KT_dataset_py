import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_csv("/kaggle/input/appliances-energy-prediction/KAG_energydata_complete.csv")
df.info()
df.describe()
df.head()
sns.distplot(df['Appliances'])
df = df.drop(labels='date', axis=1)
train, test = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)
test, val = train_test_split(df, test_size=0.5, shuffle=True, random_state=42)
class CustomDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(scale(df.drop(labels='Appliances', axis=1)).astype(np.float32)).to(device)
        self.y = torch.tensor(df['Appliances'].values.astype(np.float32)).to(device)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return {'X': self.X[index], 'y': self.y[index]}
trainD = CustomDataset(train.reset_index(drop=True))
testD = CustomDataset(test.reset_index(drop=True))
valD = CustomDataset(val.reset_index(drop=True))
trainDL = DataLoader(trainD, batch_size=32, shuffle=True, num_workers=2)
testDL = DataLoader(testD, batch_size=32, num_workers=2)
valDL = DataLoader(valD, batch_size=32, num_workers=2)
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.input_layer = nn.Linear(27, 80)
        self.hidden1 = nn.Linear(80, 40)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.batchnorm1 = nn.BatchNorm1d(40)
        self.hidden2 = nn.Linear(40, 12)
        self.output = nn.Linear(12, 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.batchnorm1(self.sigmoid(self.hidden1(x)))
        x = self.dropout(self.sigmoid(self.hidden2(x)))
        x = self.output(x)
        return x
model = FeedForwardNet()
model = model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
def trainer(epochs, trainDL, valDL, model, loss_function, optimizer):
    for epoch in range(epochs):
        for i, data in enumerate(trainDL):
            model.train()
            output = model(data['X'])
            t_loss = loss_function(output, data['y'].view(-1, 1))
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            
            v_loss = 0
            with torch.no_grad():
                model.eval()
                for j, data in enumerate(valDL):
                    loss = loss_function(model(data['X']), data['y'].view(-1, 1))
                    v_loss += loss.item()
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Training Loss: {str(round(t_loss.item(), 2))}, Validation Loss: {str(round(v_loss/j, 2))}")
trainer(5, trainDL, valDL, model, loss_function, optimizer)
def tester(testDL, model, loss_function):
    model.eval()
    total_loss = 0
    for i, data in enumerate(testDL):
        loss = loss_function(model(data['X']), data['y'].view(-1, 1))
        total_loss += loss.item()
    print(f"Total Loss: {total_loss/i}")
df = pd.read_csv("/kaggle/input/diabetes/diabetes.csv")
df.info()
df.describe()
df.head()
train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=42)
class CustomDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(scale(df.drop(labels='Outcome', axis=1)).astype(np.float32)).to(device)
        self.y = torch.tensor(df['Outcome'].values, dtype=torch.long).to(device)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return {'X': self.X[index], 'y': self.y[index]}
trainD = CustomDataset(train.reset_index(drop=True))
testD = CustomDataset(test.reset_index(drop=True))
valD = CustomDataset(val.reset_index(drop=True))
trainDL = DataLoader(trainD, batch_size=16, shuffle=True)
valDL = DataLoader(valD, batch_size=16)
testDL = DataLoader(testD, batch_size=16)
class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.input = nn.Linear(8, 20)
        self.hidden1 = nn.Linear(20, 6)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(6)
        self.output = nn.Linear(6, 2)
        
    def forward(self, x):
        x = self.input(x)
        x = self.batchnorm(self.sigmoid(self.hidden1(x)))
        x = self.output(x)
        return x
model = FFN()
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
def trainer(epochs, trainDL, valDL, model, optimizer, loss_function):
    for epoch in range(epochs):
        for i, data in enumerate(trainDL):
            model.train()
            output = model(data['X'])
            t_loss = loss_function(output, data['y'])
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                v_loss = 0
                model.eval()
                for j, data in enumerate(valDL):
                    loss = loss_function(model(data['X']), data['y'])
                    v_loss += loss.item()
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Training Loss: {str(round(t_loss.item(), 2))}, Validation Loss: {str(round(v_loss/j, 2))}")
trainer(10, trainDL, valDL, model, optimizer, loss_function)
def tester(model, testDL):
    model.eval()
    total = 0
    correct = 0
    for i, data in enumerate(testDL):
        output = model(data['X'])
        values, indices = torch.max(output.data, 1)
        total += data['y'].size(0)
        correct += (indices == data['y']).sum().item()
    print(f"Accuracy: {(correct/total)*100}")
tester(model, testDL)
