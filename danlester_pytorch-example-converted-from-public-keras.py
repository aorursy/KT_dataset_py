import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stats

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit



import pyarrow.parquet as pq

import math

import os

import gc

from tqdm import tqdm_notebook as tqdm, tnrange



print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pq.read_pandas('../input/train.parquet').to_pandas()
mdtrain = pd.read_csv('../input/metadata_train.csv')
train.info()
mdtrain.head()

def feature_extractor(x, n_part=1000):

    length = len(x)

    n_feat = 7

    pool = np.int32(np.ceil(length/n_part))

    output = np.zeros((n_part, n_feat))

    for j, i in enumerate(range(0,length, pool)):

        if i+pool < length:

            k = x[i:i+pool]

        else:

            k = x[i:]

        output[j, 0] = np.mean(k, axis=0) #mean

        output[j, 1] = np.min(k, axis=0) #min

        output[j, 2] = np.max(k, axis=0) #max

        output[j, 3] = np.std(k, axis=0) #std

        output[j, 4] = np.median(k, axis=0) #median

        output[j, 5] = stats.skew(k, axis=0) #skew

        output[j, 6] = stats.kurtosis(k, axis=0) # kurtosis

    return output
X = []

y = []

for i in tqdm(mdtrain.signal_id):

    idx = mdtrain.loc[mdtrain.signal_id==i, 'signal_id'].values.tolist()

    y.append(mdtrain.loc[mdtrain.signal_id==i, 'target'].values)

    X.append(feature_extractor(train.iloc[:, idx].values, n_part=400))
X = np.array(X).reshape(-1, X[0].shape[0], X[0].shape[1])

X = np.transpose(X, [0,2,1]) # Make X shape (batch size, channels, time steps) 

# because channels/time-steps are different in Keras



y = np.array(y).reshape(-1,1)
X.shape, y.shape
del train; gc.collect()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)

(train_idx, val_idx) = next(sss.split(X, y))



X_train, X_val = X[train_idx], X[val_idx]

y_train, y_val = y[train_idx], y[val_idx]
X_train.shape, X_val.shape, y_train.shape, y_val.shape
scalers = {} # This code actually looked wrong for Keras but right for Pytorch already!

for i in range(X_train.shape[1]):

    scalers[i] = MinMaxScaler(feature_range=(-1, 1))

    X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :]) 



for i in range(X_val.shape[1]):

    X_val[:, i, :] = scalers[i].transform(X_val[:, i, :]) 

def confusion(prediction, truth):

    """ Returns the confusion matrix for the values in the `prediction` and `truth`

    tensors, i.e. the amount of positions where the values of `prediction`

    and `truth` are

    - 1 and 1 (True Positive)

    - 1 and 0 (False Positive)

    - 0 and 0 (True Negative)

    - 0 and 1 (False Negative)

    """



    confusion_vector = torch.as_tensor(prediction, dtype=torch.float32) / torch.as_tensor(truth, dtype=torch.float32)

    # Element-wise division of the 2 tensors returns a new tensor which holds a

    # unique value for each case:

    #   1     where prediction and truth are 1 (True Positive)

    #   inf   where prediction is 1 and truth is 0 (False Positive)

    #   nan   where prediction and truth are 0 (True Negative)

    #   0     where prediction is 0 and truth is 1 (False Negative)



    true_positives = torch.sum(confusion_vector == 1).item()

    false_positives = torch.sum(confusion_vector == float('inf')).item()

    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()

    false_negatives = torch.sum(confusion_vector == 0).item()



    return true_positives, false_positives, true_negatives, false_negatives
def matthews(TP, FP, TN, FN):

    nom = TP*TN - FP*FN

    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    return nom/denom

    
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
class VSBNet1(nn.Module):

    

    def __init__(self):

        super(VSBNet1, self).__init__()

        

        self.conv1 = nn.Conv1d(in_channels=7, out_channels=64, kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)

        

        self.mp1 = nn.MaxPool1d(2, padding=1)

        

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=20, kernel_size=4)

        self.conv4 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=4)

        

        #self.gap1 = nn.AdaptiveMaxPool1d(20)

        self.gap1 = nn.AvgPool1d(192)

        

        self.do1 = nn.Dropout(0.2)

        

        # Flatten

        

        self.lin1 = nn.Linear(20,32)

        self.do2 = nn.Dropout(0.5)

        

        self.lin2 = nn.Linear(32,8)

        self.do3 = nn.Dropout(0.5)



        self.lin3 = nn.Linear(8,1)

        

        def init_weights(m):

            # Conv1d defaults to kaiming just like Keras does

            if type(m) == nn.Linear:

                # Keras defaults Dense to glorot_uniform (which is also called xavier uniform)

                # Whereas Pytorch default for Linear is kaiming

                torch.nn.init.xavier_uniform_(m.weight)

                

        self.apply(init_weights)

        

        

    def forward(self,x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        

        x = self.mp1(x)

        

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))



        x = self.gap1(x)

        

        x = self.do1(x)

        

        x = x.view(x.shape[0],-1)

        

        x = torch.tanh(self.lin1(x))

        x = self.do2(x)

        

        x = torch.tanh(self.lin2(x))

        x = self.do3(x)

        

        x = self.lin3(x) # Leave sigmoid for the loss function

            

        return x

    
net = VSBNet1()
print(net)
NUM_EPOCHS = 30

BS = 16
X_train.shape, y_train.shape
trainloader = DataLoader(list(zip(X_train,y_train)), batch_size=BS, shuffle=False, num_workers=2)
criterion = torch.nn.BCEWithLogitsLoss() #pos_weight=torch.Tensor([1.0,1.2])) - pos_weight seems to work differently on latest pytorch

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
net.train()

    

for t in tnrange(NUM_EPOCHS, desc='Epochs'):

    

    #running_loss = 0.0

    for i, (X_batch, y_target_batch) in tqdm(enumerate(trainloader), total=len(trainloader)):

              

        # Forward pass: Compute predicted y by passing x to the model

        y_pred = net(X_batch.float())

        # y_pred = net(X.view(X.shape[0], 1, X.shape[1]))



        # Compute and print loss

        loss = criterion(y_pred, y_target_batch.float())





        # Zero gradients, perform a backward pass, and update the weights.

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



net.eval()

y_preds = net(torch.Tensor(X_train).float()).detach()

loss = criterion(y_preds, torch.Tensor(y_train).float()).detach(); loss
# Number of target==0, no of target==1

(y_preds <= 0).sum().item(), (y_preds > 0).sum().item()
net.eval()

y_val_preds = net(torch.Tensor(X_val).float()).detach()

loss_val = criterion(y_val_preds, torch.Tensor(y_val).float()).detach(); loss_val
(y_val_preds <= 0).sum().item(), (y_val_preds > 0).sum().item()
c_train = confusion(y_preds > 0, y_train); c_train
matthews(*c_train)
c_val = confusion(y_val_preds > 0, y_val); c_val
matthews(*c_val)
mdtest = pd.read_csv('../input/metadata_test.csv')
mdtest.head()
start_test = mdtest.signal_id.min()

end_test = mdtest.signal_id.max()
X_test = []



pool_test = 2000



for start_col in tqdm(range(start_test, end_test + 1, pool_test)):

    end_col = min(start_col + pool_test, end_test + 1)

    test = pq.read_pandas('../input/test.parquet',

                          columns=[str(c) for c in range(start_col, end_col)]).to_pandas()



    for i in tqdm(test.columns, desc=str(start_test)):

        X_test.append(feature_extractor(test[i].values, n_part=400))

        #test.drop([i], axis=1, inplace=True); gc.collect()

    del test; gc.collect()
X_test = np.array(X_test).reshape(-1, X_test[0].shape[0], X_test[0].shape[1])

X_test = np.transpose(X_test, [0,2,1]) # Make X shape (batch size, channels, time steps) 

# because channels/time-steps are different in Keras
for i in range(X_test.shape[1]):

    X_test[:, i, :] = scalers[i].transform(X_test[:, i, :]) 
testloader = DataLoader(X_test, batch_size=100, shuffle=False, num_workers=2)
net.eval()



y_test_preds = None

for i, X_test_batch in tqdm(enumerate(testloader), total=len(testloader)):

    y_test_preds_batch = net(torch.as_tensor(X_test_batch, dtype=torch.float32)).detach()

    if y_test_preds is None:

        y_test_preds = y_test_preds_batch

    else:

        y_test_preds = torch.cat([y_test_preds, y_test_preds_batch])

        
y_test_classes = y_test_preds > 0
submission = pd.read_csv('../input/sample_submission.csv')

submission['signal_id'] = mdtest.signal_id.values

submission['target'] = y_test_classes.data.numpy().astype(int)

submission.to_csv('submission.csv', index=False)
(y_test_classes <= 0).sum().item(), (y_test_classes > 0).sum().item()