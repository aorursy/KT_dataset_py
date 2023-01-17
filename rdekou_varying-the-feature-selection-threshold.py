import numpy as np, pandas as pd, os

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))

preds = np.zeros(len(test))



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

    test3 = sel.transform(test2[cols])

    

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

       

    #if i%64==0: print(i)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('QDA scores CV =',round(auc,5))
# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof_b = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.1).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

    

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_b[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

       

    #if i%64==0: print(i)

        

# PRINT CV AUC

auc_b = roc_auc_score(train['target'],oof_b)

print('QDA scores CV =',round(auc_b,5))
val = [] 

for i in range(512):



  auc = roc_auc_score(train.loc[train['wheezy-copper-turtle-magic']==i]['target'], oof[train.loc[train['wheezy-copper-turtle-magic']==i].index])

  auc_b = roc_auc_score(train.loc[train['wheezy-copper-turtle-magic']==i]['target'], oof_b[train.loc[train['wheezy-copper-turtle-magic']==i].index])

  

  if auc_b > auc:

   print(i, 'QDA scores difference CV =',round(auc_b-auc,5))

   val.append(i)
len(val)
# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    

    if i in val: 

        sel = VarianceThreshold(threshold=1.1).fit(train2[cols])

        train3 = sel.transform(train2[cols])

        test3 = sel.transform(test2[cols])

    else:    

        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

        train3 = sel.transform(train2[cols])

        test3 = sel.transform(test2[cols])

    

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

       

    #if i%64==0: print(i)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('QDA scores CV =',round(auc,5))