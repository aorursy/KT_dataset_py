## Import the packages required for this notebook

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline
import os
## Loading the data from the Kaggle environment

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load Training data
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.head()
# Loading test data
test_df = pd.read_csv('/kaggle/input/titanic/test.csv') 
test_df.head()
## Analysis train data

print(train_df.shape)
print(train_df.dtypes)
# describe the train dataframe
train_df.describe()
## describe the test dataset

test_df.describe()
print("Missing Age values: ",train_df['Age'].isnull().sum())


average_age = train_df['Age'].mean()
print("Average Age on the existing passengers :",average_age)



### filling the missing age values with average age.
train_df['Age'] = train_df['Age'].fillna(average_age)
test_df['Age'] = test_df['Age'].fillna(average_age)
print(train_df['Embarked'].value_counts())
print("Missing Values: ", train_df['Embarked'].isnull().sum())


## using the missing Embarked values to default as "Q". This will try to balance out the value in S.

train_df['Embarked'] = train_df['Embarked'].fillna("Q")
test_df['Embarked'] = train_df['Embarked'].fillna("Q")

print(train_df['Embarked'].value_counts())
train_df['Pclass'].value_counts()  ## No missing values, PClass looks good in distribution.
train_df['Sex'].value_counts() ## no null values, features looks ok.
## creating features on train data

train_df['family_size'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['is_alone'] = np.where(train_df['family_size'] <= 1, 1, 0)
train_df['is_kid'] = np.where(train_df['Age'] <= 16, 1, 0)


## creating features on test data
test_df['family_size'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['is_alone'] = np.where(test_df['family_size'] <= 1, 1, 0)
test_df['is_kid'] = np.where(test_df['Age'] <= 16, 1, 0)
### Converting the string values to categorical numbers - on training data

train_df['Sex_cat'] = train_df['Sex'].map({'male':0, 'female':1}) ## categorical value for sex
train_df['Embarked_cat'] = train_df['Embarked'].map({'S':0,'C':1,'Q':2}) ## categorical value for Embarked

test_df['Sex_cat'] = test_df['Sex'].map({'male':0, 'female':1}) ## categorical value for sex
test_df['Embarked_cat'] = test_df['Embarked'].map({'S':0,'C':1,'Q':2}) ## categorical value for Embarked
train_df.head() ### sampling the dataframe after the transformations applied
to_drop_cols = ['Sex','Embarked','Cabin','Ticket','Parch','SibSp','Name','Fare']

for cols in to_drop_cols:
    print("Dropping ",cols)
    train_df.drop(cols, axis=1, inplace=True)
    test_df.drop(cols,axis=1, inplace=True)
train_df.head()
## Selecting the training features.

X = train_df[['Pclass','Age','family_size','is_alone','Sex_cat','Embarked_cat','is_kid']]
y = train_df['Survived']

## selecting the features for the test predictions. Note: The features are the same as in the training data.
X_test_df = test_df[['Pclass','Age','family_size','is_alone','Sex_cat','Embarked_cat','is_kid']]
X.head()
## we will use scikit-learn to split the data. Note: X and y are based of the Training dataset.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33) ## 0.2 represents 80/20 split

print(X_train.shape)
## converting splitted training dataset

X_train = torch.FloatTensor(X_train.values)
y_train = torch.LongTensor(y_train.values)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.LongTensor(y_test.values)


## converting the test dataset.
X_test_df = torch.FloatTensor(X_test_df.values)
X_train.shape ## notice it is now of type torch.
## define the model
## the model uses 2 hidden layers with 200, 100 neurons

class Model(nn.Module):
    
    def __init__(self, num_features=7, h1=200, h2=100, out_features=2):
        super().__init__()
        
        self.input = nn.Linear(num_features,h1) ## fully connected linear unit
        self.h1_layer = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self, x):
        
        x = nn.functional.relu(self.input(x)) 
        x = nn.functional.relu(self.h1_layer(x))
        x = nn.functional.relu(self.out(x)) ## ReLu activation unit for binary classification.
        
        return x
## Instantiate the model.

model = Model()
model
criterion = nn.CrossEntropyLoss() ## Using Cross Entropy Loss.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  ## Adam optimizer with learning rate = 0.001
trainloader = DataLoader(X_train, batch_size=10, shuffle=True)
testloader = DataLoader(X_test, batch_size=10, shuffle=False)
epochs = 200
losses = []

for i in range(epochs):
    i+=1
    
    optimizer.zero_grad() 
    
    y_predict = model.forward(X_train) ## predicting
    loss = criterion(y_predict, y_train) ## calculating the loss using the loss function defined above.
    
    losses.append(loss)
    
    if i%10 == 0:  ## printing out the epoch loss result after every 10 iterations.
        print(f'Epoch: {i} \t Loss: {loss:0.8f}')
        
    
    loss.backward()
    optimizer.step()
### plotting the losses

plt.plot(range(epochs), losses)
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
correct = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val_test = model.forward(data)
        
        #print(f'Row: {i} \t {y_val_test} \t {y_val_test.argmax()} \t {y_test[i]}')
        
        if y_val_test.argmax() == y_test[i]:
            correct +=1

print(f'Total Correct Predictions: {correct} out of {len(X_test)}')
print(f'Accuracy : {100 * (correct/len(X_test))}')
predict_test_df = test_df.copy() ## making a copy of the test dataframe.

with torch.no_grad():
    test_predictions = model(X_test_df) ## using the model to make the predictions.
    pred_max, pred_max_label = test_predictions.topk(1, dim=1)
    
    predict_test_df['Survived'] = pred_max_label.numpy().ravel()
    
    predict_test_df[['PassengerId','Survived']].to_csv('submission.csv',index=False) ## as per the competition rule.
torch.save(model.state_dict(),'titanic_model.pt') ## this only saves the model state.