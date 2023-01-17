import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.utils import shuffle

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



df_train = pd.read_csv('../input/titanic/train.csv')

df_test  = pd.read_csv('../input/titanic/test.csv')

df_sub   = pd.read_csv('../input/titanic/gender_submission.csv')
df_train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

df_test.drop( ['Name','Ticket','Cabin'],axis=1,inplace=True)



sex = pd.get_dummies(df_train['Sex'],drop_first=True)

embark = pd.get_dummies(df_train['Embarked'],drop_first=True)

df_train = pd.concat([df_train,sex,embark],axis=1)



df_train.drop(['Sex','Embarked'],axis=1,inplace=True)



sex = pd.get_dummies(df_test['Sex'],drop_first=True)

embark = pd.get_dummies(df_test['Embarked'],drop_first=True)

df_test = pd.concat([df_test,sex,embark],axis=1)



df_test.drop(['Sex','Embarked'],axis=1,inplace=True)



df_train.fillna(df_train.mean(),inplace=True)

df_test.fillna(df_test.mean(),inplace=True)



Scaler1 = StandardScaler()

Scaler2 = StandardScaler()



train_columns = df_train.columns

test_columns  = df_test.columns



df_train = pd.DataFrame(Scaler1.fit_transform(df_train))

df_test  = pd.DataFrame(Scaler2.fit_transform(df_test))



df_train.columns = train_columns

df_test.columns  = test_columns



features = df_train.iloc[:,2:].columns.tolist()

target   = df_train.loc[:, 'Survived'].name



X_train = df_train.iloc[:,2:].values

y_train = df_train.loc[:, 'Survived'].values
import torch

import torch.nn as nn

from torch.nn import functional as F

from torch.autograd import Variable
#thank you very much https://www.kaggle.com/mburakergenc/ttianic-minimal-pytorch-mlp

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(8, 512)

        self.fc2 = nn.Linear(512, 512)

        self.fc3 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(0.2)

        

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)

        return x

model = Net()

print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#thank you very much https://www.kaggle.com/mburakergenc/ttianic-minimal-pytorch-mlp



batch_size = 64

n_epochs = 500

batch_no = len(X_train) // batch_size



train_loss = 0

train_loss_min = np.Inf

for epoch in range(n_epochs):

    for i in range(batch_no):

        start = i*batch_size

        end = start+batch_size

        x_var = Variable(torch.FloatTensor(X_train[start:end]))

        y_var = Variable(torch.LongTensor(y_train[start:end])) 

        

        optimizer.zero_grad()

        output = model(x_var)

        loss = criterion(output,y_var)

        loss.backward()

        optimizer.step()

        

        values, labels = torch.max(output, 1)

        num_right = np.sum(labels.data.numpy() == y_train[start:end])

        train_loss += loss.item()*batch_size

    

    train_loss = train_loss / len(X_train)

    if train_loss <= train_loss_min:

        print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))

        torch.save(model.state_dict(), "model.pt")

        train_loss_min = train_loss

    

    if epoch % 200 == 0:

        print('')

        print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {}".format(epoch+1, train_loss,num_right / len(y_train[start:end]) ))

print('Training Ended! ')
X_test = df_test.iloc[:,1:].values

X_test_var = Variable(torch.FloatTensor(X_test), requires_grad=False) 

with torch.no_grad():

    test_result = model(X_test_var)

values, labels = torch.max(test_result, 1)

survived = labels.data.numpy()
submission = pd.DataFrame({'PassengerId': df_sub['PassengerId'], 'Survived': survived})

submission.to_csv('submission.csv', index=False)