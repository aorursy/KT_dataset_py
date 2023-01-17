import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
data_path = Path('/kaggle/input/titanic/')

train_path = data_path / 'train.csv'
test_path = data_path / 'test.csv'
train_data = pd.read_csv(str(train_path))
test_data = pd.read_csv(str(test_path))
train_data.head()
y = train_data['Survived']
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
train_data.isnull().sum()
# Fill missing age values with average of all ages

age_average = train_data['Age'].mean()
train_data['Age'] = train_data['Age'].fillna(age_average)
# Fill missing embarked with most common values

most_common = train_data['Embarked'].value_counts().index[0]
train_data['Embarked'] = train_data['Embarked'].fillna(most_common)
train_data.isnull().sum()
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})

# Normalize numeric columns

col_data = train_data['Fare']
train_data['Fare'] = (col_data - col_data.min())/(col_data.max() - col_data.min())

col_data = train_data['Age']
train_data['Age'] = (col_data - col_data.min())/(col_data.max() - col_data.min())

col_data = train_data['Parch']
train_data['Parch'] = (col_data - col_data.min())/(col_data.max() - col_data.min())

col_data = train_data['SibSp']
train_data['SibSp'] = (col_data - col_data.min())/(col_data.max() - col_data.min())
train_data = pd.get_dummies(train_data, columns=['Embarked'], prefix='emb_')
train_data = pd.get_dummies(train_data, columns=['Pclass'], prefix='pclass_')
train_data.head()
# Train NN
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_shape, 7)
        self.linear2 = nn.Linear(7, 3)
        self.linear3 = nn.Linear(3, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        
        x = self.linear3(x)        
        x = self.sigmoid(x)
        
        return x
    
input_data = torch.Tensor(train_data.to_numpy())
input_y = torch.Tensor(y.to_numpy())
input_y = input_y.unsqueeze(1)
loss_func = nn.BCELoss()
model = Model(input_data.shape[1])
def predict(data_input, training=False):
    if training:
        model.train()
    else:
        model.eval()
    y_pred = model(data_input)
    if not training:
        y_pred = (y_pred >= 0.5).float()
        
    return y_pred
def validate(data_input, truth_y):
    y_pred = predict(data_input, training=False)
    total = data_input.shape[0]
    correct = (y_pred == truth_y).sum().item()
    acc = correct / total
    return acc
optimizer = torch.optim.SGD(model.parameters(), lr=.1)
EPOCHS = 1000000
for epoch in range(1, EPOCHS+1):
    model.train()
    y_pred = predict(input_data, training=True)
    loss = loss_func(y_pred, input_y)
    acc = validate(input_data, input_y)
    if epoch % (EPOCHS // 10) == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():5f}, Acc: {acc:5f}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
test_data = pd.read_csv(str(test_path))
test_data.isnull().sum()
test_passenger_id = test_data['PassengerId']
test_data = test_data.drop(columns=['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1, errors='ignore')
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
# Preprocess

test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})

# Normalize numeric columns

col_data = test_data['Fare']
test_data['Fare'] = (col_data - col_data.min())/(col_data.max() - col_data.min())

col_data = test_data['Age']
test_data['Age'] = (col_data - col_data.min())/(col_data.max() - col_data.min())

col_data = test_data['Parch']
test_data['Parch'] = (col_data - col_data.min())/(col_data.max() - col_data.min())

col_data = test_data['SibSp']
test_data['SibSp'] = (col_data - col_data.min())/(col_data.max() - col_data.min())

# Encode to one-hot
test_data = pd.get_dummies(test_data, columns=['Embarked'], prefix='emb_')
test_data = pd.get_dummies(test_data, columns=['Pclass'], prefix='pclass_')
test_data.head()
test_input_x = torch.Tensor(test_data.to_numpy())
test_data.isnull().sum()
results = predict(test_input_x).squeeze()
results_df = pd.DataFrame({'PassengerId':test_passenger_id, 'Survived':list(results.numpy().astype(np.int32))})
results_df
results_df.to_csv('output.csv', index=False)
