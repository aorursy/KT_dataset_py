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



from sklearn import preprocessing

from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from sklearn import metrics

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torch.utils.data import Dataset, DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



from contextlib import contextmanager

import time

import gc

notebookstart = time.time()



@contextmanager

def timer(name):

    """

    Time Each Process

    """

    t0 = time.time()

    yield

    print('\n[{}] done in {} Minutes'.format(name, round((time.time() - t0)/60,2)))
seed = 50

debug = False



if debug:

    nrow = 5000

else:

    nrow = None
with timer("Load"):

    PATH = "/kaggle/input/cat-in-the-dat-ii/"

    train = pd.read_csv(PATH + "train.csv", index_col = 'id', nrows = nrow)

    test = pd.read_csv(PATH + "test.csv", index_col = 'id', nrows = nrow)

    submission_df = pd.read_csv(PATH + "sample_submission.csv")

    [print(x.shape) for x in [train, test, submission_df]]



    traindex = train.index

    testdex = test.index



    y_var = train.target.copy()

    print("Target Distribution:\n",y_var.value_counts(normalize = True).to_dict())



    df = pd.concat([train.drop('target',axis = 1), test], axis = 0)

    del train, test, submission_df
with timer("FE 1"):

    drop_cols=["bin_0"]



    # Split 2 Letters; This is the only part which is not generic and would actually require data inspection

    df["ord_5a"]=df["ord_5"].str[0]

    df["ord_5b"]=df["ord_5"].str[1]

    drop_cols.append("ord_5")



    xor_cols = []

    nan_cols = []

    for col in df.columns:

        # NUll Values

        tmp_null = df.loc[:,col].isnull().sum()

        if tmp_null > 0:

            print("{} has {} missing values.. Filling".format(col, tmp_null))

            nan_cols.append(col)

            if df.loc[:,col].dtype == "O":

                df.loc[:,col].fillna("NAN", inplace=True)

            else:

                df.loc[:,col].fillna(-1, inplace=True)

        

        # Categories that do not overlap

        train_vals = set(df.loc[traindex, col].unique())

        test_vals = set(df.loc[testdex, col].unique())

        

        xor_cat_vals=train_vals ^ test_vals

        if xor_cat_vals:

            df.loc[df[col].isin(xor_cat_vals), col]="xor"

            print("{} has {} xor factors, {} rows".format(col, len(xor_cat_vals),df.loc[df[col] == 'xor',col].shape[0]))

            xor_cols.append(col)





    # One Hot Encode None-Ordered Categories

    ordinal_cols=['ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5a', 'day', 'month']

    X_oh=df[df.columns.difference(ordinal_cols)]

    oh1=pd.get_dummies(X_oh, columns=X_oh.columns, drop_first=True, sparse=True)

    ohc1=oh1.sparse.to_coo()
from sklearn.base import TransformerMixin

from itertools import repeat

import scipy



class ThermometerEncoder(TransformerMixin):

    """

    Assumes all values are known at fit

    """

    def __init__(self, sort_key=None):

        self.sort_key = sort_key

        self.value_map_ = None

    

    def fit(self, X, y=None):

        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}

        return self

    

    def transform(self, X, y=None):

        values = X.map(self.value_map_)

        

        possible_values = sorted(self.value_map_.values())

        

        idx1 = []

        idx2 = []

        

        all_indices = np.arange(len(X))

        

        for idx, val in enumerate(possible_values[:-1]):

            new_idxs = all_indices[values > val]

            idx1.extend(new_idxs)

            idx2.extend(repeat(idx, len(new_idxs)))

            

        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")

            

        return result
other_classes = ["NAN", 'xor']



with timer("Thermometer Encoder"):

    thermos=[]

    for col in ordinal_cols:

        if col=="ord_1":

            sort_key=(other_classes + ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']).index

        elif col=="ord_2":

            sort_key= (other_classes + ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']).index

        elif col in ["ord_3", "ord_4", "ord_5a"]:

            sort_key=str

        elif col in ["day", "month"]:

            sort_key=int

        else:

            raise ValueError(col)



        enc=ThermometerEncoder(sort_key=sort_key)

        thermos.append(enc.fit_transform(df[col]))
ohc=scipy.sparse.hstack([ohc1] + thermos).tocsr()

display(ohc)



X_sparse = ohc[:len(traindex)]

test_sparse = ohc[len(traindex):]



print(X_sparse.shape)

print(test_sparse.shape)



del ohc; gc.collect()
# Train Test Split

X_train, X_valid, y_train, y_valid = train_test_split(X_sparse, y_var, test_size=0.3)



[print(table.shape) for table in [X_train, y_train, X_valid, y_valid]];
class TabularDataset(torch.utils.data.Dataset):

    def __init__(self, data, y=None):

        self.n = data.shape[0]

        self.y = y

        self.X = data



    def __len__(self):

        return self.n



    def __getitem__(self, idx):

        if self.y is not None:

            return [self.X[idx].toarray(), self.y.astype(float).values[idx]]

        else:

            return [self.X[idx].toarray()]
train_dataset = TabularDataset(data = X_train, y = y_train)

valid_dataset = TabularDataset(data = X_valid, y = y_valid)

submission_dataset = TabularDataset(data = test_sparse, y = None)
batch_size = 4096



train_loader = torch.utils.data.DataLoader(train_dataset, 

                                           batch_size=batch_size, 

                                           shuffle=True)



val_loader = torch.utils.data.DataLoader(valid_dataset, 

                                           batch_size=batch_size, 

                                           shuffle=False)



submission_loader = torch.utils.data.DataLoader(submission_dataset, 

                                           batch_size=batch_size, 

                                           shuffle=False)
next(iter(train_loader))
class Net(nn.Module):

    def __init__(self, dropout = .15):

        super().__init__()

        self.dropout = dropout

        

        self.fc1 = nn.Linear(X_sparse.shape[1], 64)

        self.d1 = nn.Dropout(p=self.dropout)

        self.fc2 = nn.Linear(64, 128)

        self.d2 = nn.Dropout(p=self.dropout)

        self.fc3 = nn.Linear(128, 64)

        self.d3 = nn.Dropout(p=self.dropout)

        self.fc4 = nn.Linear(64, 1)

        self.out_act = nn.Sigmoid()



    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = self.d1(x)

        x = F.relu(self.fc2(x))

        x = self.d2(x)

        x = F.relu(self.fc3(x))

        x = self.d3(x)

        x = self.fc4(x)

        x = self.out_act(x)

        return x
net = Net()

net.to(device)



learning_rate = 0.1

# https://github.com/ncullen93/torchsample/blob/master/README.md

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

scheduler = ReduceLROnPlateau(optimizer, min_lr = 0.00001, mode='min', factor=0.1, patience=2, verbose=True)



EPOCHS = 50



criterion = nn.BCELoss()

# optimizer = optim.Adam(net.parameters(), lr=0.0001)



nn_output = []

patience = 0

min_val_loss = np.Inf



for epoch in range(EPOCHS): # 3 full passes over the data

    epoch_loss = 0

    train_metric_score = 0

    net.train()

    

    for data in train_loader:  # `data` is a batch of data

        X, y = data[0].squeeze(1), data[1]  # X is the batch of features, y is the batch of targets.

        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.

        output = net(X.to(device).float()).squeeze()  # pass in the reshaped batch

        tloss = criterion(output, y.to(device))  # calc and grab the loss value

        tloss.backward()  # apply this loss backwards thru the network's parameters

        optimizer.step()  # attempt to optimize weights to account for loss/gradients 

        

        epoch_loss += tloss.item()

        score = metrics.roc_auc_score(y, output.detach().cpu().numpy())

        train_metric_score += score

    

    # Evaluation with the validation set

    net.eval() # eval mode

    val_loss = 0

    val_metric_score = 0

    with torch.no_grad():

        for data in val_loader:

            X, y = data[0].squeeze(1), data[1]

            

            preds = net(X.to(device).float()).squeeze() # get predictions

            vloss = criterion(preds.float(), y.to(device)) # calculate the loss

            

            score = metrics.roc_auc_score(y, preds.detach().cpu().numpy())

            val_metric_score += score

            val_loss += vloss.item()

    

    final_val_loss = val_loss/len(val_loader)

    tmp_nn_output = [epoch + 1,EPOCHS,

                     epoch_loss/len(train_loader),

                     train_metric_score/len(train_loader),

                     final_val_loss,

                     val_metric_score/len(val_loader)

                     ]

    nn_output.append(tmp_nn_output)

    

    # Early Stopping

    scheduler.step(final_val_loss)

    

    # Print the loss and accuracy for the validation set

    print('Epoch [{}/{}] train loss: {:.4f} train metric: {:.4f} valid loss: {:.4f} val metric: {:.4f}'

        .format(*tmp_nn_output))

    

    if min_val_loss > round(final_val_loss,4) :

        min_val_loss = round(final_val_loss,4)

        patience = 0

    else:

        patience += 1

    

    if patience > 6:

        print("Early Stopping..")

        break
pd_results = pd.DataFrame(nn_output,

    columns = ['epoch','total_epochs','train_loss','train_metric','valid_loss','valid_metric']

                         )

display(pd_results)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].plot(pd_results['epoch'],pd_results['valid_loss'], label='validation_loss')

axes[0].plot(pd_results['epoch'],pd_results['train_loss'], label='train_loss')

# axes[0].plot(pd_results['epoch'],pd_results['test_loss'], label='test_loss')

axes[0].set_title("Loss")



axes[0].legend()



axes[1].plot(pd_results['epoch'],pd_results['valid_metric'], label='Val')

axes[1].plot(pd_results['epoch'],pd_results['train_metric'], label='Train')

# axes[1].plot(pd_results['epoch'],pd_results['test_acc'], label='test_acc')

axes[1].set_title("Roc_AUC Score")

axes[1].legend()

plt.show()
net.eval() # Safety first

predictions = torch.Tensor().to(device) # Tensor for all predictions



# Go through the test set, saving the predictions in... 'predictions'

for X in submission_loader:

#     preds = net(X)

    preds = net(X[0].to(device).float()).squeeze()

    predictions = torch.cat((predictions, preds))
submission = pd.DataFrame({'id': testdex, 'target': predictions.cpu().detach().numpy()})

submission.to_csv('submission.csv', index=False)

submission.head()