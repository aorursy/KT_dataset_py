# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df = pd.from_csv("/kaggle/input/titanic/train.csv")
train_df.head()
train_df['Survived'].value_counts()
from torch.utils.data import Dataset, DataLoader, random_split

class TitanicDataset(Dataset):


    def __init__(self, csv_file):
        

        df = pd.read_csv(csv_file)
        
        drop_columns = ['Name', 'Ticket', 'Cabin']
        
        df = df.drop(drop_columns, axis=1)

        # Grouping variable names
        self.categorical = ["Embarked", "Sex"]
        self.numerical =  list(set(df.columns) - set(self.categorical) - set(["Survived"]))
        self.target = "Survived"

        # one hot encoding categorical
        self.titanic = df
        
         # one hot encoding categorical
        for category in self.categorical:
            new_columns = pd.get_dummies(self.titanic[category])
            self.titanic = pd.concat([self.titanic, new_columns],axis=1)
            
        self.titanic.drop(self.categorical,axis=1,inplace=True)

        
        #scaling numerical
        from sklearn.preprocessing import StandardScaler
 
        self.titanic[self.numerical] = self.titanic[self.numerical].fillna(0)
        scaler = StandardScaler()
        self.titanic[self.numerical] = scaler.fit_transform(self.titanic[self.numerical])
        
        self.target = "Survived"

        self.X = self.titanic.drop(self.target, axis=1)
        print(f'{len(self.X.columns)} columns in dataset')
        self.y = self.titanic[self.target]

    def __len__(self):
        return len(self.titanic)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]
from typing import Tuple
import torch
from torch.nn import Linear
from torch.nn.functional import leaky_relu

class MLP(torch.nn.Module):

    def __init__(self, in_num: int, hidden_layers: Tuple[int, int], out_num: int):
        super(MLP, self).__init__()
        self.linear1 = Linear(in_num, hidden_layers[0])
        self.linear2 = Linear(hidden_layers[0], hidden_layers[1])
        self.linear3 = Linear(hidden_layers[1], out_num)

    def forward(self, x):
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        return self.linear3(x)

dataset = TitanicDataset("/kaggle/input/titanic/train.csv")
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = random_split(dataset, [train_size, test_size])

use_cuda = torch.cuda.is_available()
dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

train_loader = DataLoader(trainset, batch_size=100, shuffle=True, **dataloader_kwargs)
test_loader = DataLoader(testset, batch_size=test_size, shuffle=False, **dataloader_kwargs)

dataset.X.head()
# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'Training size {train_size} | Test size {test_size} | device {device} | gpu {use_cuda}')

model = MLP(11, (512, 512), 1).to(device)
print(model)

loss_function = torch.nn.BCEWithLogitsLoss()

learning_rate = .01

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
epochs = 1500
model.train()
for epoch in range(epochs):
    avg_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        output = model(inputs.float())
        loss = loss_function(output, labels.float().unsqueeze(-1))
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
        
    if epoch % 100 == 0:
        print(f'Epoch {epoch} Loss: {avg_loss}')
test_data = iter(test_loader)
threshold = .5
model.eval() #letting pytorch know this is evaluation phase
test_loss = 0
accuracy = 0
with torch.no_grad(): # don't calculate gradients
    for x, y in test_loader:
        inputs = x.to(device)
        y = y.to(device)
        output = model(inputs.float())
        test_loss += loss_function(output.float(), y.float().unsqueeze(1)).item()
        preds = torch.sigmoid(output)
        predicted_vals = torch.tensor([int(t.item() > threshold) for t in preds])
        accuracy += predicted_vals.eq(y).sum().item()

test_loss /= len(test_loader.dataset)
accuracy /= len(test_loader.dataset)
print(f'Test set: Loss: {test_loss} | Accuracy: {accuracy}')
predict_dataset = TitanicDataset("/kaggle/input/titanic/train.csv")
predict_loader = DataLoader(predict_dataset, len(predict_dataset.titanic), shuffle=False, **dataloader_kwargs)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text

for i, (inputs, labels) in enumerate(predict_loader):
    with torch.no_grad():
        output = model(inputs.float())
        preds = torch.sigmoid(output)
        predicted_vals = torch.tensor([int(t.item() > threshold) for t in preds])

    decision_tree = DecisionTreeClassifier(random_state=0, max_depth = 3)
    decision_tree.fit(inputs, predicted_vals)


feature_names = list(set(predict_dataset.titanic.columns) - set(["Survived"]))
tree_rules = export_text(decision_tree, feature_names=feature_names)

print(tree_rules)
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
tree.plot_tree(decision_tree, filled=True,feature_names=feature_names) 
plt.show()
feat_importance = decision_tree.tree_.compute_feature_importances(normalize=False)
feature = (dict(zip(feature_names, feat_importance)))

feature
