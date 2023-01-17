import numpy as np, pandas as pd, os

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import roc_auc_score

from scipy.special import expit

from tqdm import tqdm

import matplotlib.pyplot as plt





train = pd.read_csv('../input/instant-gratification/train.csv')

test = pd.read_csv('../input/instant-gratification/test.csv')
import gc

import multiprocessing

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from scipy.special import logit, expit as sigmoid



import torch

from torch.nn import functional as F

from torch.utils.data import DataLoader, Dataset

torch.multiprocessing.set_sharing_strategy('file_system')



import warnings

warnings.filterwarnings("ignore")
def cross_entropy(y, p_logits):

    p = F.softmax(p_logits, dim=1)

    return F.cross_entropy(p, y, reduction="mean")



class LinearModel(torch.nn.Module):

    def __init__(self, n_features):

        super(LinearModel, self).__init__()

        self.linear = torch.nn.Linear(n_features, 2)



    def forward(self, x):

        return self.linear(x)

    

class DistillationDataset(Dataset):

    def __init__(self, X, y=None, t=None):

        self.X = X

        self.y = y

        if t is None:

            self.t = None

        else:

            #self.t = t

            t = np.clip(t, -100, 0)

            spread = np.diff(np.quantile(t, [0.1, 0.9]))

            self.t = (t / spread).astype(np.float32)#(t.max() - t.min())

            #t = (1-1e-6)*(t - 0.5) + 0.5

            #t = np.clip(np.vstack([np.log(1-t), np.log(t)]).T, -50, 50)

            #self.t = 2*(t - t.min()) / (t.max() - t.min()) - 1.0            

            #print(spread)

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, index):

        X = self.X[index]

        y = 1.0 if self.y is None else self.y[index]

        t = 1.0 if self.t is None else self.t[index]

        return index, X, y, t    
batch_size=64

lr=0.0005

n_epochs=100

l1_penalty=0.01

l2_penalty=0.5

device = torch.device("cpu")



# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

train_preds = np.zeros(len(train))

final_preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in range(512):

    

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    if len(test2) > 0:

        test_data = DistillationDataset(test2[cols].values.astype(np.float32))

        test_loader = DataLoader(test_data, batch_size)        



    # STRATIFIED K-FOLD

    avg_score = []

    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle=True)

    for train_index, valid_index in skf.split(train2, train2['target']):              



        # create datasets

        train_data = DistillationDataset(train2.loc[train_index, cols].values.astype(np.float32),

                                         train2.loc[train_index, 'target'].values.astype(np.int64))

        train_loader = DataLoader(train_data, batch_size, shuffle=True)



        # create model

        model = LinearModel(len(cols)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)



        # train model

        train_score = []

        for epoch in range(n_epochs):

            model.train()

            for _, X_batch, y_batch, t_logits in train_loader:

                s_logits = model(X_batch.to(device))

                ce_loss = cross_entropy(y_batch.to(device), s_logits.to(device))

                l1_reg = sum(x.abs().sum() for n,x in model.named_parameters() if "weight" in n)

                loss = ce_loss + l1_penalty*l1_reg                

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



        # inference

        model.eval()

        valid_data = DistillationDataset(train2.loc[valid_index, cols].values.astype(np.float32))

        valid_loader = DataLoader(valid_data, batch_size)    

        for indices, X_batch, _, _ in valid_loader:

            train_preds[idx1[valid_index][indices]] = F.softmax(model(X_batch.to(device)), dim=1)[:,1].cpu().detach()               



        if len(test2) > 0:

            for indices, X_batch, _, _ in test_loader:

                final_preds[idx2[indices]] = F.softmax(model(X_batch.to(device)), dim=1)[:,1].cpu().detach() / skf.n_splits



# PRINT CV AUC

auc = roc_auc_score(train['target'], train_preds)

print('CV scores =',round(auc,5))
sub = pd.read_csv('../input/instant-gratification/sample_submission.csv')

sub['target'] = final_preds

sub.to_csv('submission.csv',index=False)