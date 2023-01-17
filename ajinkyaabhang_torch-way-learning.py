import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
from sklearn import model_selection



x = df.drop('Outcome', axis =1).values

y = df['Outcome'].values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state = 42)
import torch

import torch.nn as nn

import torch.nn.functional as f
# creating tensors

X_train = torch.FloatTensor(X_train)

X_test = torch.FloatTensor(X_test)

Y_train = torch.LongTensor(Y_train)

Y_test = torch.LongTensor(Y_test)
class ANN_model(nn.Module):

    def __init__(self, input_features=8, hidden1=20, hidden2=20, output_features=2):

        super().__init__()

        self.fully_connected1=nn.Linear(input_features,hidden1)

        self.fully_connected2=nn.Linear(hidden1,hidden2)

        self.out=nn.Linear(hidden2,output_features)

    def forward(self,x):

        x=f.relu(self.fully_connected1(x))

        x=f.relu(self.fully_connected2(x))

        x=self.out(x)

        return x
# instantiate ANN model



torch.manual_seed(42)

model=ANN_model()
model.parameters
#backward propogation

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10000

final_losses =[]

for i in range(epochs):

    i = i+1

    y_pred = model.forward(X_train)

    loss = loss_function(y_pred, Y_train)

    final_losses.append(loss)

    if i%10 == 1:

        print("Epoch Number : {} and the loss is {}".format(i, loss.item()))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
### plot the loss function

import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(range(epochs),final_losses)

plt.ylabel('Loss')

plt.xlabel('Epoch')
#### Prediction In X_test data

predictions=[]

with torch.no_grad():

    for i,data in enumerate(X_test):

        y_pred=model(data)

        predictions.append(y_pred.argmax().item())
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,predictions)

cm
from sklearn.metrics import accuracy_score

score=accuracy_score(Y_test,predictions)

score