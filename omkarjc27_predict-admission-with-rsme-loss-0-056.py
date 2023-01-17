import pandas as pd
import numpy as np
df=pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv', sep=',')
dataset = df.values
dataset = dataset[:,1:] # Removed serial number column
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = dataset[:,:7]
y = dataset[:,7]

scaler = preprocessing.StandardScaler().fit(X)
scaler.transform(X)
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0) # 70% training and 20% test

positives = X[y>=0.5,:]
print("% +ve",len(positives)*100/len(dataset))
negatives = X[y<0.5,:]
print("% -ve",len(negatives)*100/len(dataset))
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

for i in range(7):
    n, bins, patches = plt.hist(negatives[:,i], 5, facecolor='red', alpha=0.5)
    n, bins, patches = plt.hist(positives[:,i], 5, facecolor='green', alpha=0.5)
    plt.xlabel(df.columns[i+1])
    plt.show()
import torch

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
class Feedforward(torch.nn.Module):
        def __init__(self):
            super(Feedforward, self).__init__()
            self.a1 = torch.nn.Linear(7,10)
            self.a2 = torch.nn.Linear(10,10)
            self.a3 = torch.nn.Linear(10, 1)
            
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()        
        def forward(self, x):
            z1 = self.relu(self.a1(x))
            z2 = self.relu(self.a2(z1))
            z3 = self.sigmoid(self.a3(z2))
            return z3
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,y_pred,y_real):
        return torch.sqrt(self.mse(y_pred,y_real))
model = Feedforward()
criterion = RMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.006)
model.train()
epoch = 100000
loss_arr = []
for epoch in range(epoch):
    optimizer.zero_grad()    # Forward pass
    y_pred = model(X_train)    # Compute Loss
    loss = criterion(y_pred.squeeze(), y_train)
    if (epoch+1)%5000 == 0:
        print('Epoch {}: train loss: {}'.format((epoch+1), loss.item()))    # Backward pass
        loss_arr.append(float(loss.item()))
    loss.backward()
    optimizer.step()
print('Done.')
plt.plot(list(range(len(loss_arr))),loss_arr,color='green')
plt.ylabel('loss')
plt.xlabel('iter')
plt.show()
from sklearn import metrics
model.eval()
y_pred = model(X_test)
after_train = criterion(y_pred.squeeze(), y_test) 
print('Loss on last 100 datapoints ' , after_train.item())
print("R^2 Score:",metrics.r2_score(y_test.detach().numpy(), y_pred.detach().numpy()))