import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn import neural_network

from sklearn.svm import NuSVC

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, TruncatedSVD



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



oof = np.zeros(len(train))

preds = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



for i in range(512):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    #data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))

    data2 = StandardScaler().fit_transform(TruncatedSVD(n_components=40, random_state=4).fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

    if i%15==0: print(i)



#print(roc_auc_score(train['target'], oof))

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv', index=False)