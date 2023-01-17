import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # For Normalization
device = torch.device("cuda")
torch.manual_seed(484)
torch.cuda.manual_seed_all(484)
x_train = pd.read_csv('/kaggle/input/lol-prediction/lol.x_train.csv', index_col=0)
y_train = pd.read_csv('/kaggle/input/lol-prediction/lol.y_train.csv', index_col=0)
x_test = pd.read_csv('/kaggle/input/lol-prediction/lol.x_test.csv', index_col=0)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
print(x_train.shape)
print(x_test.shape)
scaler = StandardScaler() # Normalizer
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # Not Fit Transform!

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test).to(device)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
train_set = TensorDataset(x_train, y_train)
data_loader = DataLoader(dataset=train_set,
                         batch_size=10000,
                         shuffle=True)
# DNN 모델 구축
linear1 = torch.nn.Linear(48, 32).to(device)
linear2=torch.nn.Linear(32,64).to(device)
linear3 = torch.nn.Linear(64, 1).to(device)

relu = torch.nn.LeakyReLU()
sigmoid = torch.nn.Sigmoid()
dropout=torch.nn.Dropout(p=0.5)

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)


model = torch.nn.Sequential(linear1, relu, dropout,
                            linear2, relu, dropout,
                            linear3, sigmoid)

model
! pip3 install adamp
from adamp import AdamP
cost = torch.nn.BCELoss().to(device)
optimizer = AdamP(model.parameters(), lr=1e-2, weight_decay=1e-2)
epochs = 120
for epoch in range(1, epochs+1):
    avg_cost = 0
    total_batch = len(data_loader)

    for x, y in data_loader: 
        # batch loop
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        hypothesis = model(x)
        cost_val = cost(hypothesis, y)
        cost_val.backward()
        optimizer.step()

        avg_cost += cost_val
    
    avg_cost /= total_batch

    if epoch % 10 == 1 or epoch == epochs:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, avg_cost.item()))
with torch.no_grad(): # Don't Calculate Gradient
    model.eval()
    x_test = x_test.to(device)

    pred = model(x_test)
pred[pred>=0.5] = 1.0
pred[pred<=0.5] = 0.0
pred = pred.detach().cpu().numpy()
pred = pred.astype(np.uint32)
id=np.array([i for i in range(pred.shape[0])]).reshape(-1, 1).astype(np.uint32)
result=np.hstack([id, pred])

submit = pd.DataFrame(result, columns=['id', 'blueWins'])
submit.to_csv('submit.csv', index=False)