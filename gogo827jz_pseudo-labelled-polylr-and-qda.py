import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
print('Loading Train')

train = pd.read_csv('../input/train.csv')

print('Loading Test')

test = pd.read_csv('../input/test.csv')

print('Finish')
oof = np.zeros(len(train))

preds = np.zeros(len(test))



oof_QDA = np.zeros(len(train))

preds_QDA = np.zeros(len(test))



cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



for i in range(512):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # Adding quadratic polynomial features can help linear model such as Logistic Regression learn better

    poly = PolynomialFeatures(degree=2)

    sc = StandardScaler()

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = sc.fit_transform(poly.fit_transform(VarianceThreshold(threshold=1.5).fit_transform(data[cols])))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=1.5).fit_transform(data[cols])

    train4 = data2[:train2.shape[0]]; test4 = data2[train2.shape[0]:]

    

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = LogisticRegression(solver='saga',penalty='l2',C=0.01,tol=0.001)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

        clf_QDA = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf_QDA.fit(train4[train_index,:],train2.loc[train_index]['target'])

        oof_QDA[idx1[test_index]] = clf_QDA.predict_proba(train4[test_index,:])[:,1]

        preds_QDA[idx2] += clf_QDA.predict_proba(test4)[:,1] / skf.n_splits

        

    if i%64==0:

        print(i, 'LR oof auc : ', round(roc_auc_score(train['target'][idx1], oof[idx1]), 5))

        print(i, 'QDA oof auc : ', round(roc_auc_score(train['target'][idx1], oof_QDA[idx1]), 5))
# INITIALIZE VARIABLES

test['target'] = preds

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for k in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    

    # ADD PSEUDO LABELED DATA

    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()

    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 

    train2p = pd.concat([train2p,test2p],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    

     # FEATURE SELECTION AND ADDING POLYNOMIAL FEATURES

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train3p = sel.transform(train2p[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])   

    poly = PolynomialFeatures(degree=2).fit(train3p)

    train3p = poly.transform(train3p)

    train3 = poly.transform(train3)

    test3 = poly.transform(test3)

    sc2 = StandardScaler()

    train3p = sc2.fit_transform(train3p)

    train3 = sc2.transform(train3)

    test3 = sc2.transform(test3)

        

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof

        

        # MODEL AND PREDICT WITH LR

        clf = LogisticRegression(solver='saga',penalty='l2',C=0.01,tol=0.001)

        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof[idx1[test_index3]] = clf.predict_proba(train3[test_index3,:])[:,1]

        preds[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

    if k%64==0:  

        print(k, 'LR2 oof auc : ', round(roc_auc_score(train['target'][idx1], oof[idx1]), 5))
# INITIALIZE VARIABLES

test['target'] = preds_QDA

oof_QDA2 = np.zeros(len(train))

preds_QDA2 = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for k in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    

    # ADD PSEUDO LABELED DATA

    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()

    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 

    train2p = pd.concat([train2p,test2p],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train3p = sel.transform(train2p[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

        

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof

        

        # MODEL AND PREDICT WITH QDA

        clf_QDA2 = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf_QDA2.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof_QDA2[idx1[test_index3]] = clf_QDA2.predict_proba(train3[test_index3,:])[:,1]

        preds_QDA2[test2.index] += clf_QDA2.predict_proba(test3)[:,1] / skf.n_splits

       

    if k%64==0:

        print(k, 'QDA2 oof auc : ', round(roc_auc_score(train['target'][idx1], oof_QDA2[idx1]), 5))
print('LR auc: ', round(roc_auc_score(train['target'], oof),5))

print('QDA auc: ', round(roc_auc_score(train['target'], oof_QDA2),5))
w_best = 0

oof_best = oof_QDA2

for w in np.arange(0,0.55,0.001):

    oof_blend = w*oof+(1-w)*oof_QDA2

    if (roc_auc_score(train['target'], oof_blend)) > (roc_auc_score(train['target'], oof_best)):

        w_best = w

        oof_best = oof_blend

        print(w_best)

print('best weight: ', w_best)

print('auc_best: ', round(roc_auc_score(train['target'], oof_best), 5))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = w_best*preds + (1-w_best)*preds_QDA2

sub.to_csv('submission.csv', index=False)

sub.head()