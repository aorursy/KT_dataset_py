import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")
normal_data = data[data["Class"] == 0].drop(["Class"], axis=1)
anormal_data = data[data["Class"] == 1].drop(["Class"], axis=1)
X_train = normal_data.values[0:283823]
X_normal_test = normal_data.values[283823:]
X_anormal_test = anormal_data.values
# 標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(30, 10)
        self.fc2 = nn.Linear(10, 30)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, data):

        self.data = data

        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        
        return out_data
trainset = Mydatasets(X_train_std)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 100, shuffle = True, num_workers = 2)
model = Model()
learning_rate = 0.0001
epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.SmoothL1Loss()
for epoch in range(epochs):
    train_loss = 0
    model.train()
    model.double()
    for data in trainloader:
        
        res = model(data)
        
        loss = criterion(res, data)
        
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(train_loss/len(trainloader))
X_normal_test_pred = model(torch.Tensor(sc.transform(X_normal_test)).double())
X_anormal_test_pred = model(torch.Tensor(sc.transform(X_anormal_test)).double())
normal_diff = X_normal_test_pred.detach().numpy() - X_normal_test
anormal_diff = X_anormal_test_pred.detach().numpy() - X_anormal_test
X = np.array(list(normal_diff) + list(anormal_diff))
y = [0] * 492 + [1] * 492
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = SVC(gamma=0.01)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
