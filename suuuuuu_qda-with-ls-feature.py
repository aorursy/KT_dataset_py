import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from tqdm import tqdm_notebook

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.semi_supervised import LabelSpreading

from sklearn.mixture import GaussianMixture

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
train = pd.read_csv('../input/train.csv')

train_t = train.copy()

test = pd.read_csv('../input/test.csv')
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

reg_best = [0.5,

0.2,

0.3,

0.2,

0.5,

0.1,

0.1,

0.2,

0.3,

0.5,

0.2,

0.4,

0.1,

0.3,

0.1,

0.4,

0.3,

0.2,

0.2,

0.5,

0.1,

0.4,

0.4,

0.1,

0.5,

0.4,

0.1,

0.4,

0.4,

0.1,

0.1,

0.3,

0.4,

0.1,

0.5,

0.2,

0.3,

0.1,

0.1,

0.5,

0.5,

0.5,

0.3,

0.5,

0.4,

0.1,

0.1,

0.1,

0.5,

0.5,

0.5,

0.1,

0.3,

0.1,

0.1,

0.4,

0.2,

0.3,

0.1,

0.1,

0.5,

0.2,

0.4,

0.1,

0.1,

0.1,

0.1,

0.4,

0.2,

0.1,

0.1,

0.5,

0.4,

0.1,

0.3,

0.2,

0.4,

0.1,

0.3,

0.5,

0.1,

0.5,

0.1,

0.5,

0.1,

0.1,

0.4,

0.5,

0.4,

0.2,

0.1,

0.1,

0.4,

0.5,

0.2,

0.5,

0.5,

0.4,

0.1,

0.5,

0.5,

0.3,

0.5,

0.2,

0.4,

0.4,

0.1,

0.4,

0.4,

0.1,

0.1,

0.5,

0.5,

0.5,

0.1,

0.2,

0.4,

0.1,

0.4,

0.5,

0.5,

0.5,

0.2,

0.2,

0.2,

0.5,

0.1,

0.1,

0.3,

0.5,

0.3,

0.1,

0.4,

0.1,

0.3,

0.1,

0.2,

0.5,

0.5,

0.1,

0.1,

0.1,

0.4,

0.1,

0.5,

0.5,

0.5,

0.1,

0.5,

0.5,

0.1,

0.5,

0.5,

0.2,

0.4,

0.2,

0.1,

0.5,

0.3,

0.5,

0.2,

0.4,

0.4,

0.5,

0.2,

0.3,

0.1,

0.1,

0.5,

0.1,

0.5,

0.5,

0.5,

0.5,

0.1,

0.5,

0.4,

0.1,

0.4,

0.3,

0.4,

0.4,

0.3,

0.1,

0.4,

0.4,

0.2,

0.5,

0.4,

0.4,

0.2,

0.1,

0.2,

0.5,

0.5,

0.1,

0.5,

0.3,

0.4,

0.5,

0.1,

0.5,

0.5,

0.5,

0.1,

0.1,

0.3,

0.2,

0.5,

0.1,

0.5,

0.5,

0.4,

0.1,

0.5,

0.1,

0.5,

0.1,

0.3,

0.3,

0.1,

0.1,

0.1,

0.4,

0.3,

0.1,

0.1,

0.4,

0.3,

0.3,

0.4,

0.5,

0.2,

0.1,

0.5,

0.5,

0.4,

0.4,

0.3,

0.1,

0.1,

0.5,

0.1,

0.1,

0.1,

0.1,

0.3,

0.3,

0.2,

0.1,

0.5,

0.4,

0.3,

0.1,

0.3,

0.1,

0.2,

0.4,

0.5,

0.3,

0.1,

0.1,

0.3,

0.3,

0.4,

0.4,

0.2,

0.5,

0.1,

0.5,

0.3,

0.1,

0.2,

0.5,

0.1,

0.1,

0.5,

0.4,

0.1,

0.5,

0.5,

0.5,

0.3,

0.2,

0.4,

0.5,

0.4,

0.3,

0.1,

0.4,

0.3,

0.2,

0.2,

0.1,

0.4,

0.4,

0.1,

0.2,

0.1,

0.5,

0.3,

0.2,

0.1,

0.2,

0.3,

0.2,

0.5,

0.4,

0.5,

0.5,

0.1,

0.1,

0.4,

0.3,

0.3,

0.4,

0.3,

0.2,

0.5,

0.4,

0.1,

0.1,

0.4,

0.1,

0.1,

0.5,

0.4,

0.1,

0.4,

0.5,

0.3,

0.2,

0.5,

0.4,

0.4,

0.5,

0.1,

0.1,

0.5,

0.5,

0.5,

0.1,

0.5,

0.1,

0.5,

0.2,

0.1,

0.1,

0.1,

0.5,

0.5,

0.4,

0.5,

0.1,

0.3,

0.5,

0.5,

0.3,

0.5,

0.1,

0.3,

0.1,

0.4,

0.3,

0.5,

0.5,

0.5,

0.4,

0.2,

0.5,

0.5,

0.5,

0.5,

0.1,

0.1,

0.1,

0.5,

0.4,

0.3,

0.1,

0.5,

0.5,

0.2,

0.3,

0.5,

0.5,

0.1,

0.1,

0.1,

0.5,

0.3,

0.1,

0.4,

0.1,

0.1,

0.5,

0.5,

0.4,

0.1,

0.5,

0.4,

0.2,

0.5,

0.1,

0.4,

0.1,

0.1,

0.1,

0.4,

0.2,

0.1,

0.2,

0.2,

0.5,

0.4,

0.1,

0.1,

0.1,

0.4,

0.5,

0.4,

0.1,

0.1,

0.1,

0.1,

0.1,

0.1,

0.3,

0.5,

0.3,

0.5,

0.5,

0.5,

0.5,

0.5,

0.3,

0.5,

0.5,

0.1,

0.5,

0.1,

0.1,

0.1,

0.1,

0.2,

0.1,

0.5,

0.5,

0.1,

0.1,

0.4,

0.3,

0.1,

0.2,

0.1,

0.1,

0.1,

0.3,

0.5,

0.2,

0.1,

0.3,

0.2,

0.4,

0.4,

0.2,

0.1,

0.3,

0.1,

0.1,

0.4,

0.1,

0.2,

0.4,

0.5,

0.3,

0.1,

0.1,

0.5,

0.5,

0.5,

0.5,

0.5,

0.1,

0.1,

0.1,

0.1,

0.4,

0.1,

0.4,

0.2,

0.1,

0.1,

0.4,

0.1,

0.5,

0.2,

0.1,

0.1,

0.3,

0.5,

0.1,

0.5,

0.5,

0.1,

0.1,

0.2,

0.1,

0.1,

0.1,

0.2,

0.3]
oof = np.zeros(len(train))

preds = np.zeros(len(test))



for i in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])



    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = QuadraticDiscriminantAnalysis(reg_best[i])

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')

auc = roc_auc_score(train_t['target'], oof)

print(f'AUC: {auc:.5}')
idx_t0 = train[train['target']== 0].index

idx_t1 = train[train['target']== 1].index

import matplotlib.pyplot as plt

plt.hist(oof[idx_t0],bins=100,alpha=0.5)

plt.hist(oof[idx_t1],bins=100,alpha=0.5)

plt.title('Final Test.csv predictions')   

plt.ylim([0,5000])

plt.show()
train.loc[oof > 0.99, 'target'] = 1

train.loc[oof < 0.01, 'target'] = 0
oof_ls = np.zeros(len(train)) 

pred_te_ls = np.zeros(len(test))

for k in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    test2['target']=-1

    #test2p = test2[ (test2['target']<=0.05) | (test2['target']>=0.95) ].copy()

    #test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

    #test2p.loc[ test2p['target']<0.5, 'target' ] = 0 

    train2p = pd.concat([train2,test2],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    

    #merging train2p with full test

#     train3p = pd.concat([train2p,test2],axis=0)

#     train3p.reset_index(drop=True,inplace=True)

    

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train4p = sel.transform(train2p[cols])

    train4 = sel.transform(train2[cols])

    test4 = sel.transform(test2[cols])

    

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train4p, train2p['target']):

        test_index3 = test_index[ test_index<len(train4) ] 

        

        clf = LabelSpreading(gamma=0.01,kernel='rbf', max_iter=10,n_jobs=-1)

        clf.fit(train4p[train_index,:],train2p.loc[train_index]['target'])

        oof_ls[idx1[test_index3]] = clf.predict_proba(train4[test_index3,:])[:,1]

        pred_te_ls[test2.index] += clf.predict_proba(test4)[:,1] / skf.n_splits

auc = roc_auc_score(train['target'],oof_ls)

print('CV for LabelSpreading =',round(auc,5))   
idx_t0 = train[train_t['target']== 0].index

idx_t1 = train[train_t['target']== 1].index

import matplotlib.pyplot as plt

plt.hist(oof_ls[idx_t0],bins=100,alpha=0.5)

plt.hist(oof_ls[idx_t1],bins=100,alpha=0.5)

plt.title('Final Test.csv predictions')   

# plt.ylim([0,5000])

plt.show()
oof = np.zeros(len(train))

preds = np.zeros(len(test))



for i in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    train4 = np.hstack([train3,np.array([oof_ls[idx1]]).T])

    test4 = np.hstack([test3,np.array([pred_te_ls[idx2]]).T])

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train4, train2['target']):



        clf = QuadraticDiscriminantAnalysis(reg_best[i])

        clf.fit(train4[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train4[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test4)[:,1] / skf.n_splits



auc = roc_auc_score(train_t['target'], oof)

print(f'AUC: {auc:.5}')
idx_t0 = train_t[train_t['target']== 0].index

idx_t1 = train_t[train_t['target']== 1].index

import matplotlib.pyplot as plt

plt.hist(oof[idx_t0],bins=100,alpha=0.5)

plt.hist(oof[idx_t1],bins=100,alpha=0.5)

plt.title('Final Test.csv predictions')   

plt.ylim([0,5000])

plt.show()
import matplotlib.pyplot as plt

plt.hist(preds,bins=100)

plt.title('Final Test.csv predictions')   

plt.show()
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)