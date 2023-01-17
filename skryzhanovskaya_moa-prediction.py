import numpy as np 

import pandas as pd 



import os



import seaborn as sns



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib

%config InlineBackend.figure_format = 'svg'



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# matplotlib.style.use('seaborn') 



if torch.cuda.is_available():

    device = 'cuda'

else:

    device = 'cpu'
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_features.head()
train_targets_scored.head()
train_features.shape, train_targets_scored.shape, train_targets_scored.shape
train_features.isnull().sum().sum()
train_features.sig_id.nunique()
(train_features.sig_id != train_targets_scored.sig_id).sum()
train_features.set_index('sig_id', inplace=True)

test_features.set_index('sig_id', inplace=True)

train_targets_scored.set_index('sig_id', inplace=True)

train_targets_nonscored.set_index('sig_id', inplace=True)
train_targets_scored.sum(axis=1).value_counts()
train_targets_scored.sum(axis=0)
fig, ax = plt.subplots()

fig.set_figwidth(10)

fig.set_figheight(5)



plt.scatter(np.arange(train_targets_scored.shape[1]), train_targets_scored.sum(axis=0)) 

plt.grid(True)

plt.ylabel('Положительные исходы')

plt.xlabel('Номер целевой переменной')

plt.xticks(np.arange(train_targets_scored.shape[1])[::10])

plt.show()
train_targets_scored.loc[:, train_targets_scored.sum(axis=0) > 600]
from collections import Counter

moa_types = Counter([name.split('_')[-1] for name in train_targets_scored.columns])
moa_types
train_features.cp_type.value_counts()
train_targets_scored.loc[train_features[train_features.cp_type == 'ctl_vehicle'].index].sum(axis=0).sum()
train_features.cp_time.value_counts()
train_features.cp_dose.value_counts()
drop_index = train_features[train_features.cp_type == 'ctl_vehicle'].index



train_features_df = train_features.drop(drop_index, axis=0)

train_features_df = train_features_df.drop('cp_type', axis=1)



train_target_df = train_targets_scored.drop(drop_index, axis=0)





drop_index = test_features[test_features.cp_type == 'ctl_vehicle'].index

test_features_df = test_features.drop(drop_index, axis=0)

test_features_df = test_features_df.drop('cp_type', axis=1)
train_features_df = pd.get_dummies(train_features_df, columns=['cp_time', 'cp_dose'], drop_first=True)

test_features_df = pd.get_dummies(test_features_df , columns=['cp_time', 'cp_dose'], drop_first=True)
X_train_all = train_features_df.values

y_train_all = train_target_df.values

X_test = test_features_df.values
scaler = StandardScaler()

X_train_all = scaler.fit_transform(X_train_all)

X_test = scaler.transform(X_test)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
train_all_dataset = TensorDataset(torch.tensor(X_train_all).float(), torch.tensor(y_train_all).float())

train_all_loader = DataLoader(train_all_dataset, batch_size=512)



train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())

val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())



train_loader = DataLoader(train_dataset, batch_size=512)

val_loader = DataLoader(val_dataset, batch_size=512)
x, y = next(iter(train_loader))

x.shape, y.shape
class FFNN(nn.Module):

    def __init__(self, input_size, output_size, dropout_rate=0.1):

        super().__init__()

        

        hidden_size = input_size // 2

        

        self.l1 = nn.Linear(input_size, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.dropout1 = nn.Dropout(dropout_rate)

        

        self.l2 = nn.Linear(hidden_size, hidden_size)

        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)

        

        self.l3 = nn.Linear(hidden_size, output_size)

        

    

    def forward(self, x):

        x = self.l1(x)

        x = self.bn1(x)

        x = self.dropout1(x)

        x = F.relu(x)

        

        x = self.l2(x)

        x = self.bn2(x)

        x = self.dropout2(x)

        x = F.relu(x)

        

        x = self.l3(x)

        

        return x
model = FFNN(input_size=875, output_size=206)
model(x).shape
def train_model(model, optimizer, loss_function, train_loader, 

                val_loader=None, scheduler=None, epochs=1):



    for epoch in range(epochs):

        running_loss = 0.0

        for n_iter, (x, y) in enumerate(train_loader):

            model.train()

            x = x.to(device)

            y = y.to(device) 

            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_function(y_pred, y)

            loss.backward()

            optimizer.step()      

            running_loss += loss.item()

        running_loss /= len(train_loader)   

        

        if val_loader is not None:

            model.eval()  

            loss = 0.0

            with torch.no_grad():

                for (x, y) in val_loader:

                    x = x.to(device)

                    y = y.to(device) 

                    y_pred = model(x)

                    loss += loss_function(y_pred, y).item()

                loss /= len(val_loader)



            print("Epoch: [{}/{}] ".format(epoch + 1, epochs),

                  "Train loss: {:.6f}".format(running_loss),

                  "Val loss: {:.6f} ".format(loss))

        else:

            print("Epoch: [{}/{}] ".format(epoch + 1, epochs),

                  "Train loss: {:.6f}".format(running_loss))

        if scheduler is not None:

            scheduler.step()     
loss_function = nn.BCEWithLogitsLoss()

model = FFNN(input_size=875, output_size=206).to(device)

optimizer = optim.Adam(lr=0.0001, params=model.parameters())
# train_model(model, optimizer, loss_function, train_loader, val_loader, epochs=100)
loss_function = nn.BCEWithLogitsLoss()

model = FFNN(input_size=875, output_size=206).to(device)

optimizer = optim.Adam(lr=0.0001, params=model.parameters())

train_model(model, optimizer, loss_function, train_all_loader, epochs=100)
def predict(model, X):

    model.eval()  

            

    with torch.no_grad():

        X = X.to(device)

        preds = model(X)

        y_pred = torch.sigmoid(preds)

    return y_pred.cpu().numpy()
y_pred = predict(model, torch.tensor(X_test).float())
submission = pd.DataFrame(np.zeros((test_features.shape[0], train_targets_scored.shape[1])),

                         index=test_features.index, columns=train_targets_scored.columns)
sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
pred_index = test_features[test_features.cp_type != 'ctl_vehicle'].index
len(pred_index)
y_pred.shape
submission.shape
submission.loc[pred_index, :] = y_pred
submission.reset_index(inplace=True)
submission
sample_submission.shape
submission.shape
submission.to_csv('/kaggle/working/submission.csv', index=False)