import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
df1 = pd.read_csv("../input/titanic/train.csv")
df1.head()
val = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']
plt.figure(figsize = (15,15))
for i in range(5):
    plt.subplot(2, 3, i+1), sns.countplot(x = val[i], hue = 'Survived', data = df1)
# Sex Encoding
binar = LabelBinarizer().fit(df1.loc[:, "Sex"])
df1["Sex"] = binar.transform(df1["Sex"])
# Embarked Encoding
df1["Embarked"] = df1["Embarked"].fillna('S')
df_Embarked = pd.get_dummies(df1.Embarked)
df1 = pd.concat([df1, df_Embarked], axis=1)

#Family
df1['Family'] = df1['SibSp'] + df1['Parch'] + 1
df1['Alone'] = df1['Family'].apply(lambda x : 0 if x>1 else 1 )
#Age
df1['Age'] = df1['Age'].fillna(-0.5)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
print(test_data.head())
# Sex Encoding
binar = LabelBinarizer().fit(test_data.loc[:, "Sex"])
test_data["Sex"] = binar.transform(test_data["Sex"])
# Embarked Encoding
test_data["Embarked"] = test_data["Embarked"].fillna('S')
df_Embarked = pd.get_dummies(test_data.Embarked)
test_data = pd.concat([test_data, df_Embarked], axis=1)
#Family
test_data['Family'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['Alone'] = test_data['Family'].apply(lambda x : 0 if x>1 else 1 )
#Age
test_data['Age'] = test_data['Age'].fillna(-0.5)
sc = StandardScaler()

features = ["Pclass","Sex", "Age", "C", "Q", "S", "Alone"]
X_train = df1[features]
Y_train = df1["Survived"]
test_df = sc.fit_transform(test_data[["Pclass","Sex", "Age", "C", "Q", "S", "Alone"]])
#print(test_df)
# Feature Scaling
X_train = sc.fit_transform(X_train)
print(X_train)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=35)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(Y_train.values).long()
y_test = torch.from_numpy(Y_val.values).long()
print(X_train.shape, X_test.shape)
class Neural_Network(nn.Module):
    def __init__(self): 
        super(Neural_Network, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(7, 100),
                nn.Dropout(0.5), #50 % probability 
                nn.ReLU(),
                torch.nn.Linear(100, 100),
                torch.nn.Dropout(0.2), #20% probability
                torch.nn.ReLU(),
                torch.nn.Linear(100, 2),
            )
             
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
   
def train_model(model, train_data, test_data, epochs=5, verbose=False):
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    # Loop over the epochs
    train_losses, test_losses = [0]*epochs, [0]*epochs
    accuracy = [0]*epochs
    for e in range(epochs):
        
        # Iterate the model, note we are passing in the
        # entire training set as a single batch
        optimizer.zero_grad()
        ps = model(train_data[0])
        loss = criterion(ps, train_data[1])
        loss.backward()
        optimizer.step()
        train_losses[e] = loss.item()

        # Compute the test stats
        with torch.no_grad():
            # Turn on all the nodes
            model.eval()
            
            # Comput test loss
            ps = model(test_data[0])
            loss = criterion(ps, test_data[1])
            test_losses[e] = loss.item()
            
            # Compute accuracy
            top_p, top_class = ps.topk(1, dim=1)
            equals      = (top_class == test_data[1].view(*top_class.shape))
            accuracy[e] = torch.mean(equals.type(torch.FloatTensor))
            
        model.train()
        
    # Print the final information
    print(f'   Accuracy  : {100*accuracy[-1].item():0.2f}%')
    print(f'   Train loss: {train_losses[-1]}')
    print(f'   Test loss : {test_losses[-1]}')
        
    # Plot the results
    fig = plt.figure(figsize=(14, 6))
    plt.subplot(211)
    plt.ylabel('Accuracy')
    plt.plot(accuracy)
    plt.subplot(212)
    plt.ylabel('Loss')
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.legend();
    return
model = Neural_Network()
train_model(model, epochs = 1000,
            train_data = (X_train,y_train), test_data = (X_test,y_test))
test_X = torch.tensor(test_df).float()
test_out = model(test_X)
_, preds = torch.max(test_out, 1)
print(len(preds))
pred = []
for i in range(len(preds)):
    pred.append([892 + i, preds[i].item()])

sub_df = pd.DataFrame(pred)
sub_df.columns = ["PassengerId", "Survived"]
sub_df.head()
sub_df.to_csv("my_submission.csv", index=False)
print("Your submission was successfully saved!")
sub_df.hist(column="Survived", bins=2, )