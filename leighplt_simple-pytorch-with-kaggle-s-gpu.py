import numpy as np
import pandas as pd
import gc

import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import TruncatedSVD, FastICA, PCA
from sklearn.random_projection import GaussianRandomProjection
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
y = np.log1p(train_df.target.values) # get target
train_df.drop(["ID", 'target'], axis=1, inplace = True)
test_df.drop(["ID"], axis=1, inplace = True)
df = pd.concat([train_df, test_df]).reset_index(drop=True)
del train_df, test_df
gc.collect();
# apply np.log1p to values
df.loc[:, :] = np.log1p(df.values)
SEED = 717
np.random.seed = SEED
# Code used from kernel:
# https://www.kaggle.com/nanomathias/linear-regression-with-elastic-net
COMPONENTS = 20

# List of decomposition methods to use
methods = [
    TruncatedSVD(n_components=COMPONENTS),
    PCA(n_components=COMPONENTS),
    GaussianRandomProjection(n_components=COMPONENTS, eps=0.1, random_state = SEED + 354),  
]

# Run all the methods
embeddings = []
for method in methods:
    name = method.__class__.__name__    
    embeddings.append(
        pd.DataFrame(method.fit_transform(df), columns=[f"{name}_{i}" for i in range(COMPONENTS)])
    )
    print(f">> Ran {name}")
    
# Put all components into one dataframe
X = pd.concat(embeddings, axis=1).reset_index(drop=True)
del embeddings, method, methods, name
gc.collect();
%%time
dropzero = []
for row in np.arange(df.shape[0]):
    dropzero.append( df.values[row, np.nonzero(df.values[row])] )

additional_df = pd.DataFrame(index=df.index) # initialize DataFrame for additional features

additional_df['mean'] = [x.mean() for x in dropzero]
additional_df['std'] = [x.std(ddof=1) for x in dropzero] 
additional_df['var'] = [x.var(ddof=1) for x in dropzero] 
additional_df['q_0_25'] = [np.percentile(x, q = 25) for x in dropzero]  
additional_df['q_0_50'] = [np.percentile(x, q = 50) for x in dropzero] 
additional_df['q_0_75'] = [np.percentile(x, q = 75) for x in dropzero]  
additional_df.fillna(additional_df.mean(axis = 1), inplace=True)
X = X.join(additional_df).fillna(0)
del additional_df
gc.collect();
# https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    print('using device: cuda')
else:
    print('using device: cpu')
# create torch model with 3 linear layers 
torchm = nn.Sequential(
        nn.Linear(X.shape[1], 1024),
        nn.Linear(1024, 512),
        nn.Linear(512, 1))
# our optimizer and function for loss
optimizer = optim.Adam(torchm.parameters(), lr=0.001)
criterion = nn.MSELoss() 
if USE_GPU and torch.cuda.is_available():
    X = torch.from_numpy(X.values).cuda()
    y = torch.from_numpy(y).view(-1,1).cuda()
    dtype = torch.cuda.FloatTensor
    torchm.cuda()
else:
    X = torch.from_numpy(X.values)
    y = torch.from_numpy(y).view(-1,1)
    dtype = torch.FloatTensor

X = Variable(X).type(dtype)
y = Variable(y).type(dtype)
# epoches for train; overfitting on train
torch.cuda.manual_seed_all(SEED)
epoches = 126

for epoch in np.arange(epoches):
    def closure():
        optimizer.zero_grad()
        out = torchm(X[:y.shape[0]])
        loss = criterion(out, y)**.5 # rescale to RMSE
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    if epoch % 25 == 0:
        print(f"RMSE for {epoch} epoch : {loss.data[0]}" )
# get predicts for test
preds = torchm(X[y.shape[0]:])
if USE_GPU and torch.cuda.is_available():
    preds = preds.cpu().data.numpy()
else:
    preds = preds.data.numpy()
# save prediction
sub = pd.read_csv('../input/sample_submission.csv')
sub.target = np.expm1(preds)
sub.to_csv('torch.csv', index=False)