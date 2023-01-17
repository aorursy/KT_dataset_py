# Imports



import jovian

import torch

import torch.nn as nn

import torch.nn.functional as F

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
iris = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')

iris.head()
mappings = {

   'Iris-setosa': 0,

   'Iris-versicolor': 1,

   'Iris-virginica': 2

}

iris['Name'] = iris['Name'].apply(lambda x: mappings[x])
X = iris.drop('Name', axis=1).values

y = iris['Name'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)

X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)

y_test = torch.LongTensor(y_test)
X_train
y_train
class ANN(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(in_features=4, out_features=16)

        self.fc2 = nn.Linear(in_features=16, out_features=12)

        self.output = nn.Linear(in_features=12, out_features=3)

 

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.output(x)

        return x
model = ANN()

model
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
%%time

epochs = 100

loss_arr = []

for i in range(epochs):

   y_hat = model.forward(X_train)

   loss = criterion(y_hat, y_train)

   loss_arr.append(loss)

 

   if i % 10 == 0:

       print(f'Epoch: {i} Loss: {loss}')

 

   optimizer.zero_grad()

   loss.backward()

   optimizer.step()
preds = []

with torch.no_grad():

   for val in X_test:

       y_hat = model.forward(val)

       preds.append(y_hat.argmax().item())

    

preds
df = pd.DataFrame({'Y': y_test, 'YHat': preds})

df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
df
accuracy = df['Correct'].sum() / len(df)

accuracy
jovian.commit()