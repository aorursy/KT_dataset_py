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





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

pseudo_preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    if len(test2) > 0:

        test3 = sel.transform(test2[cols])

    

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])

        if len(test2) > 0:

            pseudo_preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
# INITIALIZE VARIABLES

test['target'] = pseudo_preds

teacher_preds = np.zeros((len(train), 2))



# BUILD 512 SEPARATE MODELS

for k in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    

    # ADD PSEUDO LABELED DATA

    test2p = test2[ (test2['target'] <= 0.01) | (test2['target'] >= 0.99) ].copy()

    test2p.loc[ test2p['target'] >= 0.5, 'target' ] = 1

    test2p.loc[ test2p['target'] < 0.5, 'target' ] = 0 

    train2p = pd.concat([train2p,test2p],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train3p = sel.transform(train2p[cols])

    train3 = sel.transform(train2[cols])

    if len(test2) > 0:

        test3 = sel.transform(test2[cols])

        

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in qda1_preds

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3p[train_index,:], train2p.loc[train_index]['target'])

        teacher_preds[idx1[test_index3]] = clf.predict_log_proba(train3[test_index3,:])
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

from torch.optim.lr_scheduler import LambdaLR

torch.multiprocessing.set_sharing_strategy('file_system')



import warnings

warnings.filterwarnings("ignore")
def soft_cross_entropy(teacher_logits, student_logits, temperature):

    y = F.softmax(teacher_logits/temperature, dim=1)

    p = F.log_softmax(student_logits/temperature, dim=1)

    return temperature**2 * F.kl_div(p, y, reduction="batchmean")



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

            self.t = (t / spread).astype(np.float32)

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

l2_penalty=1.0

temperature=9.0

alpha=0.05

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

                                         train2.loc[train_index, 'target'].values.astype(np.int64),

                                         teacher_preds[train_index].astype(np.float32))

        train_loader = DataLoader(train_data, batch_size, shuffle=True)



        # create model

        model = LinearModel(len(cols)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)

        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 2**-(epoch // 50))



        # train model

        train_score = []

        for epoch in range(n_epochs):

            model.train()

            for _, X_batch, y_batch, t_logits in train_loader:

                s_logits = model(X_batch.to(device))

                sce_loss = soft_cross_entropy(t_logits.to(device), s_logits.to(device), temperature)

                ce_loss = cross_entropy(y_batch.to(device), s_logits.to(device))

                l1_reg = sum(x.abs().sum() for n,x in model.named_parameters() if "weight" in n)

                loss = alpha*sce_loss + (1-alpha)*ce_loss + l1_penalty*l1_reg                

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



            # update lr

            scheduler.step()

        

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
# group_list = np.random.choice(512,50,replace=False)

# group_list
# %%time

# batch_size=32

# lr=0.0005

# n_epochs=200

# l1_penalty=0.01

# l2_penalty=0.5

# temperature=9.0

# alpha=0.05

# device = torch.device("cpu")



# plt.figure(figsize=(30, 15))



# for alpha in [0, 0.01, 0.05, 0.10]:





#     # INITIALIZE VARIABLES

#     cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

#     train_preds = np.zeros(len(train))

#     final_preds = np.zeros(len(test))



#     # BUILD 512 SEPARATE MODELS

#     for model_num, i in enumerate(group_list):# # range(512):#

#         # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

#         train2 = train[train['wheezy-copper-turtle-magic']==i]

#         test2 = test[test['wheezy-copper-turtle-magic']==i]

#         idx1 = train2.index; idx2 = test2.index

#         train2.reset_index(drop=True,inplace=True)



#         if len(test2) > 0:

#             test_data = DistillationDataset(test2[cols].values.astype(np.float32))

#             test_loader = DataLoader(test_data, batch_size)        



#         # STRATIFIED K-FOLD

#         avg_score = []

#         skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

#         for train_index, valid_index in skf.split(train2, train2['target']):              



#             # create datasets

#             train_data = DistillationDataset(train2.loc[train_index, cols].values.astype(np.float32),

#                                              train2.loc[train_index, 'target'].values.astype(np.int64),

#                                              teacher_preds[train_index].astype(np.float32))

#             train_loader = DataLoader(train_data, batch_size, shuffle=True)



#             # create model

#             model = LinearModel(len(cols)).to(device)

#             optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)

#             scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 2**-(epoch // 50))



#             # train model

#             train_score = []

#             for epoch in range(n_epochs):

#                 model.train()

#                 for _, X_batch, y_batch, t_logits in train_loader:

#                     s_logits = model(X_batch.to(device))

#                     sce_loss = soft_cross_entropy(t_logits.to(device), s_logits.to(device), temperature)

#                     ce_loss = cross_entropy(y_batch.to(device), s_logits.to(device))

#                     l1_reg = sum(x.abs().sum() for n,x in model.named_parameters() if "weight" in n)

#                     loss = alpha*sce_loss + (1-alpha)*ce_loss + l1_penalty*l1_reg

#                     optimizer.zero_grad()

#                     loss.backward()

#                     optimizer.step()



#                 # inference

#                 model.eval()

#                 valid_data = DistillationDataset(train2.loc[valid_index, cols].values.astype(np.float32))

#                 valid_loader = DataLoader(valid_data, batch_size)    

#                 for indices, X_batch, _, _ in valid_loader:

#                     train_preds[idx1[valid_index][indices]] = F.softmax(model(X_batch.to(device)), dim=1)[:,1].cpu().detach()

#                 train_score.append(roc_auc_score(train['target'].values[idx1[valid_index]], train_preds[idx1[valid_index]]))

                

#                 # update lr

#                 scheduler.step()

                

#             avg_score.append(train_score)



#         plt.subplot(5, 10, model_num+1)

#         auc = roc_auc_score(train['target'].values[train_preds > 0], train_preds[train_preds > 0])

#         plt.plot(np.mean(avg_score, 0), label=f"{alpha} {auc : 0.5f}")

#         plt.legend()



# plt.show()
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = final_preds

sub.to_csv('submission.csv',index=False)