import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.svm import NuSVC

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.feature_selection import VarianceThreshold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm_notebook

from random import sample



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



oof = np.zeros(len(train))

preds = np.zeros(len(test))



cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(2).fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = QuadraticDiscriminantAnalysis(reg_param = 0.111)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

        

print(roc_auc_score(train['target'], oof))

oof_nusvc = np.zeros(len(train))

preds_nusvc = np.zeros(len(test))



for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(PCA(svd_solver='full',n_components='mle').fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        

        oof_nusvc[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds_nusvc[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

        

print(roc_auc_score(train['target'], oof_nusvc))
oof_knn = np.zeros(len(train))

preds_knn = np.zeros(len(test))

for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        k=KNeighborsClassifier(17,p=2.9)

        k.fit(train3[train_index,:],train2.loc[train_index]['target'])

        

        oof_knn[idx1[test_index]] = k.predict_proba(train3[test_index,:])[:,1]

        preds_knn[idx2] += k.predict_proba(test3)[:,1] / skf.n_splits

        

        

print(roc_auc_score(train['target'], oof_knn))
data_tr=pd.DataFrame({'qda':oof,'nusvc':oof_nusvc,'knn':oof_knn})

data_ts=pd.DataFrame({'qda':preds,'nusvc':preds_nusvc, 'knn':preds_knn})



index_trn=sample(list(data_tr.index),round(len(data_tr)*0.8))



logi1 = LogisticRegression('l2',1,.01,.05,1,solver='liblinear',max_iter=500)

logi1.fit(data_tr.loc[index_trn,:].values,train.loc[index_trn,'target'])

est_train=logi1.predict_proba(data_tr.drop(labels=index_trn,axis=0).values)[:,1]

est_tst=logi1.predict_proba(data_ts.values)[:,1]



print(roc_auc_score(train['target'], oof))

print(roc_auc_score(train['target'], oof_knn))

print(roc_auc_score(train.drop(labels=index_trn,axis=0)['target'], est_train))

print(roc_auc_score(train['target'], 0.8*oof+0.2*oof_nusvc))

print(roc_auc_score(train['target'], 0.95*oof+0.05*oof_nusvc))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = 0.8*preds+0.2*preds_knn

sub.to_csv('submission1.csv', index=False)



sub['target'] = 0.95*preds+0.05*preds_knn

sub.to_csv('submission2.csv', index=False)



sub['target'] = est_tst

sub.to_csv('submission3.csv', index=False)