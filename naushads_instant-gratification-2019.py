# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Essential Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from tqdm import tqdm_notebook as tqdm



#Scikit-Learn Helpers

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



#Models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



#Misc 

import warnings

warnings.filterwarnings("ignore")

from IPython.display import display

pd.set_option("display.max_columns", None)



scaler_type = "no"

N_SPLITS = 10
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")

train.head()
train['target'].value_counts()

# theres no class imbalance here, infact both the classes are equally represented
t_des = train.describe()

t_des
plt.hist(t_des.loc['mean'],bins=100);

#Some/one variable seems to be having mean > 250
display(t_des.loc['mean'][t_des.loc['mean'] > 250])
plt.hist(t_des.loc['std'],bins=100);

#Some/one variable seems to be having standard deviation of > ~140
# Culprit variable for abnormal mean and standard deviation

display(t_des.loc['std'][t_des.loc['std'] > 140])


t_des['wheezy-copper-turtle-magic']
plt.figure(figsize=(15,5))

plt.hist(train['wheezy-copper-turtle-magic'].values,bins=900);

# this feature doesnt follow normal distribution
def scale(train,test):

    traintest = pd.concat([train,test],axis=0,ignore_index=True).reset_index(drop=True)

    cols = [c for c in train.columns if c not in ['id','target','wheezy-copper-turtle-magic']]

    scaler = StandardScaler()

    traintest[cols] = scaler.fit_transform(traintest[cols])

    train['wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].astype('category')

    train = traintest[:train.shape[0]].reset_index(drop=True)

    test = traintest[train.shape[0]:].reset_index(drop=True)

    return train,test

if scaler_type == "standard":

    train,test = scale(train,test)
plt.figure(figsize=(15,15))



#DISTPLOT FOR FIRST 8 VARIABLES

for i in range(8):

    plt.subplot(3,3,i+1)

    sns.distplot(train.iloc[:,i+1],bins=200)

    plt.title(train.columns[i+1])

    plt.xlabel('')

#     plt.title(train.columns[i])



#DISPLOT FOR GAUSSIAN (FOR COMPARISON)

plt.subplot(3,3,9)

std = round(np.std(train.iloc[:,8]),2)

data = np.random.normal(0,std,train.shape[0])

sns.distplot(data,bins=200)

plt.title("Gaussian with mean=0,std="+str(std)+"\n("+train.columns[8]+")")

plt.xlabel('')



plt.show()
from scipy import stats

plt.figure(figsize=(15,15))



#NORMALITY PLOTS FOR FIRST 8 VARIABLES

for i in range(8):

    plt.subplot(3,3,i+1)

    stats.probplot(train.iloc[:,i+1],plot=plt)

    plt.title(train.columns[i+1])

    plt.xlabel('')



#NORMALITY PLOT FOR GAUSSIAN

plt.subplot(3,3,9)

std = round(np.std(train.iloc[:,8]),2)

data = np.random.normal(0,std,train.shape[0])

stats.probplot(data,plot=plt)

plt.title("Gaussian with mean=0,std="+str(std)+"\n("+train.columns[8]+")")

plt.xlabel('')



plt.show()
train0 = train[train['wheezy-copper-turtle-magic'] == 0]

plt.figure(figsize=(15,15))



#DISTPLOT FOR FIRST 8 VARIABLES

for i in range(8):

    plt.subplot(3,3,i+1)

    sns.distplot(train0.iloc[:,i+1],bins=10)

    plt.title(train0.columns[i+1])

    plt.xlabel('')



#DISPLOT FOR GAUSSIAN (FOR COMPARISON)

plt.subplot(3,3,9)

std0 = round(np.std(train0.iloc[:,8]),2)

data0 = np.random.normal(0,std0,train0.shape[0])

sns.distplot(data0,bins=10)

plt.title("Gaussian with mean=0,std="+str(std)+"\n("+train0.columns[8]+")")

plt.xlabel('')



plt.show()
# Normality Plots for Partial Dataset

plt.figure(figsize=(15,15))



#NORMALITY PLOTS FOR FIRST 8 VARIABLES OF PARTIAL DATASET

for i in range(8):

    plt.subplot(3,3,i+1)

    stats.probplot(train0.iloc[:,i+1],plot=plt)

    plt.title(train0.columns[i+1])

    plt.xlabel('')



#NORMALITY PLOT FOR GAUSSIAN

plt.subplot(3,3,9)

stats.probplot(data,plot=plt)

plt.title("Gaussian with mean=0,std="+str(std)+"\n("+train0.columns[8]+")")

plt.xlabel('')



plt.show()
# %%time

# ###########################################Logistic Regression Baseline########################

# features_to_use = [c for c in train.columns if c not in ['id','target']]

# # X_train = train[features_to_use]

# # y_train = train['target']

# # lr = LogisticRegression(C=1.0,solver='sag')

# # cv_score = cross_val_score(lr, X_train, y_train, scoring='roc_auc',cv=3)

# # print(cv_score)

# # print(np.mean(cv_score))



# # %%time

# kfold = KFold(n_splits=N_SPLITS,shuffle=True,random_state=42)

# oof = np.zeros(train.shape[0]) ## Out Of Fold (predictions)

# pred = 0

# for fold_,(trn_idx,val_idx) in enumerate(kfold.split(X=train,y=train['target'])):

#     print("fold {}".format(fold_+1))

#     X_train, y_train = train.iloc[trn_idx][features_to_use], train['target'].iloc[trn_idx]

#     X_val, y_val = train.iloc[val_idx][features_to_use], train['target'].iloc[val_idx]

#     lg = LogisticRegression(C=1.0,solver='sag')

#     lg.fit(X_train,y_train)

#     val_pred = lg.predict_proba(X_val)[:,1]

#     #Each row/datapoint would require a prediction on both 0 and 1. 

#     #For example datapoint1 has 80% likelihood to belong to 0, and 20% belonging to 1. the output would be (0.8,0.2). 

#     #you need to access prediciton[:,1] to get the second column if u want the prediction for 1. 

#     #In general access prediction[:,k] if you want likelihood of the k th class

#     oof[val_idx] = val_pred

#     pred += lg.predict_proba(test[features_to_use])[:,1]/N_SPLITS

#     print(roc_auc_score(y_val,val_pred))

# print(roc_auc_score(train['target'].values,oof))
plt.figure(figsize=(15,5))



# PLOT ALL ZIPPY

plt.subplot(1,2,1)

sns.distplot(train[ (train['target']==0) ]['zippy-harlequin-otter-grandmaster'], label = 't=0')

sns.distplot(train[ (train['target']==1) ]['zippy-harlequin-otter-grandmaster'], label = 't=1')

plt.title("Without interaction, zippy has neutral correlation with the target variable \n (showing all rows)")

plt.xlim((-4,4))

plt.legend()



# PLOT ZIPPY WHERE WHEEZY-MAGIC=0

plt.subplot(1,2,2)

sns.distplot(train[(train['target']==0) & (train['wheezy-copper-turtle-magic']==0)]

             ['zippy-harlequin-otter-grandmaster'],label='t=0')

sns.distplot(train[(train['target']==1) & (train['wheezy-copper-turtle-magic']==0)]

             ['zippy-harlequin-otter-grandmaster'],label='t=1')

plt.xlim(-2,2)

plt.title("With Interaction, zippy has positively correlation with target variable \n(showing only rows where 'wheezy-copper-turtle-magic'=0)")

plt.legend()



plt.show()
plt.figure(figsize=(15,5))



# PLOT ALL ZIPPY

plt.subplot(1,2,1)

sns.distplot(train[ (train['target']==0) ]['chewy-lime-peccary-fimbus'], label = 't=0')

sns.distplot(train[ (train['target']==1) ]['chewy-lime-peccary-fimbus'], label = 't=1')

plt.title("Without interaction, chewy has neutral correlation with the target variable \n (showing all rows)")

plt.xlim((-4,4))

plt.legend()



# PLOT ZIPPY WHERE WHEEZY-MAGIC=0

plt.subplot(1,2,2)

sns.distplot(train[(train['target']==0) & (train['wheezy-copper-turtle-magic']==0)]

             ['chewy-lime-peccary-fimbus'],label='t=0')

sns.distplot(train[(train['target']==1) & (train['wheezy-copper-turtle-magic']==0)]

             ['chewy-lime-peccary-fimbus'],label='t=1')

plt.xlim(-2,2)

plt.title("With Interaction, zippy has negative correlation with the target variable \n (showing only rows where 'wheezy-copper-turtle-magic'=0)")

plt.legend()



plt.show()
# %%time

# #Bojan's Way

# #################Logistic Regression with interaction##########################

# #initialize-variables

# cols = [c for c in train.columns if c not in ['id','target','wheezy-copper-turtle-magic']]

# target = train['target']

# NSPLITS = 25

# oof_lr = np.zeros(train.shape[0])

# pred_lr = np.zeros(test.shape[0])

# # 

# skf = StratifiedKFold(n_splits=NSPLITS,random_state=42,shuffle=True)



# for fold_,(train_idx, val_idx) in enumerate(skf.split(train.values,target.values)):

#     print('Fold',fold_+1)

#     train_f = train.iloc[train_idx]

#     val_f,y_val = train.iloc[val_idx],target.iloc[val_idx]

#     for i in tqdm(range(512)):

#         train2 = train_f[train_f['wheezy-copper-turtle-magic'] == i]

#         val2 = val_f[val_f['wheezy-copper-turtle-magic'] == i]

#         test2 = test[test['wheezy-copper-turtle-magic'] == i]

#         idx1 = train2.index;idx2 = val2.index;idx3 = test2.index

#         train2.reset_index(drop=True,inplace=True); val2.reset_index(drop=True,inplace=True); test2.reset_index(drop=True,inplace=True);

#         lg = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)

#         lg.fit(train2[cols],train2['target'])

#         oof_lr[idx2] = lg.predict_proba(val2[cols])[:,1]

#         pred_lr[idx3] += lg.predict_proba(test2[cols])[:,1]/NSPLITS

# #     print(roc_auc_score(y_val,oof_lr[val_idx]))

# print('LB with interactions',round(roc_auc_score(target,oof_lr),5))
# %%time

# #Chris's Way



# cols = [c for c in train.columns if c not in ['id','target','wheezy-copper-turtle-magic']]

# oof_lr = np.zeros(train.shape[0])

# pred_lr = np.zeros(test.shape[0])



# #512 models for each wheezy-copper-turtle-magic value

# for i in tqdm(range(train['wheezy-copper-turtle-magic'].nunique())):

#     train2 = train[train['wheezy-copper-turtle-magic']==i]

#     test2 = test[test['wheezy-copper-turtle-magic']==i]

#     idx1 = train2.index; idx2 = test2.index

#     train2.reset_index(drop=True,inplace=True); test2.reset_index(drop=True, inplace=True)

    

#     skf = StratifiedKFold(n_splits=25,random_state=42)

#     for fold_,(trn_idx,val_idx) in enumerate(skf.split(train2,train2['target'])):

#         X_train,y_train = train2.iloc[trn_idx], train2['target'].iloc[trn_idx]

#         X_val,y_val = train2.iloc[val_idx], train2['target'].iloc[val_idx]

#         lg = LogisticRegression(solver='liblinear',C=0.05,penalty='l1')

#         lg.fit(X_train[cols],y_train)

#         oof_lr[idx1[val_idx]] = lg.predict_proba(X_val[cols])[:,1]

#         pred_lr[idx2] += lg.predict_proba(test2[cols])[:,1]/25.0

# print('CV score-LR (with interaction)',round(roc_auc_score(train['target'],oof_lr),5))
# %%time

# cols = [c for c in train.columns if c not in ['id','target','wheezy-copper-turtle-magic']]

# oof_svm = np.zeros(train.shape[0])

# pred_svm = np.zeros(test.shape[0])

# skf = StratifiedKFold(n_splits=25,random_state=42)

# for i in tqdm(range(train['wheezy-copper-turtle-magic'].nunique())):

#     train2 = train[train['wheezy-copper-turtle-magic']==i]

#     test2 = test[test['wheezy-copper-turtle-magic']==i]

#     idx1=train2.index;idx2=test2.index

#     train2.reset_index(drop=True,inplace=True);test2.reset_index(drop=True,inplace=True)

    

#     #When building a model with only 512 observations and 255 features, 

#     #it is important to reduce the number of features to prevent overfitting.

#     #255 to ~44 features

#     sel = VarianceThreshold(threshold=1.5).fit(test2[cols])

#     train3 = sel.transform(train2[cols])

#     test3 = sel.transform(test2[cols])

    

#     for fold_, (train_idx,val_idx) in enumerate(skf.split(train3,train2['target'])):

#         svc = SVC(kernel='poly',degree=4,gamma='auto',probability=True)

#         svc.fit(train3[train_idx,:],train2['target'].iloc[train_idx])

#         oof_svm[idx1[val_idx]]=svc.predict_proba(train3[val_idx,:])[:,1]

#         pred_svm[idx2] += svc.predict_proba(test3)[:,1]/skf.n_splits

# roc = roc_auc_score(train['target'],oof_svm)

# print('CV score-SVM',round(roc,5))
# print('CV score-svm0.9',round(roc_auc_score(train['target'],0.9*oof_svm+0.1*oof_lr),5))

# print('CV score-svm0.8',round(roc_auc_score(train['target'],0.8*oof_svm+0.2*oof_lr),5))

# print('CV score-svm0.7',round(roc_auc_score(train['target'],0.7*oof_svm+0.3*oof_lr),5))

# print('CV score-svm0.6',round(roc_auc_score(train['target'],0.6*oof_svm+0.4*oof_lr),5))

# print('CV score-svm0.5',round(roc_auc_score(train['target'],0.5*oof_svm+0.5*oof_lr),5))
# # sample_submission['target'] = pred_svm

# # sample_submission.to_csv("submission_svm.csv", index=False)



# sample_submission['target'] = 0.9*pred_svm+0.1*pred_lr

# sample_submission.to_csv("submission_svm1.csv", index=False)

# sample_submission['target'] = 0.8*pred_svm+0.2*pred_lr

# sample_submission.to_csv("submission_svm2.csv", index=False)

# sample_submission['target'] = 0.7*pred_svm+0.3*pred_lr

# sample_submission.to_csv("submission_svm3.csv", index=False)

# sample_submission['target'] = 0.6*pred_svm+0.4*pred_lr

# sample_submission.to_csv("submission_svm4.csv", index=False)

# sample_submission['target'] = 0.5*pred_svm+0.5*pred_lr

# sample_submission.to_csv("submission_svm5.csv", index=False)
# %%time

# cols = [c for c in train.columns if c not in ['id','target','wheezy-copper-turtle-magic']]



# oof_qda = np.zeros(train.shape[0])

# pred_qda = np.zeros(test.shape[0])



# for i in tqdm(range(512)):

#     train2 = train[train['wheezy-copper-turtle-magic']==i]

#     test2 = test[test['wheezy-copper-turtle-magic']==i]

#     idx1=train2.index; idx2=test2.index

#     train2.reset_index(drop=True,inplace=True);test2.reset_index(drop=True,inplace=True)

    

#     data = pd.concat([pd.DataFrame(train2[cols]),pd.DataFrame(test2[cols])])

#     data0 = VarianceThreshold(1.5).fit_transform(data)

#     train3 = data0[:train2.shape[0]]; test3=data0[train2.shape[0]:]

    

#     skf = StratifiedKFold(n_splits=11,random_state=42)

#     for trn_idx,val_idx in skf.split(train3,train2['target']):

# #         clf = QuadraticDiscriminantAnalysis(0.1) #0.9649

#         clf = QuadraticDiscriminantAnalysis(0.08) #

#         clf.fit(train3[trn_idx,:],train2.loc[trn_idx]['target'])

#         oof_qda[idx1[val_idx]] = clf.predict_proba(train3[val_idx,:])[:,1]

#         pred_qda[idx2] += clf.predict_proba(test3)[:,1]/skf.n_splits

# auc = roc_auc_score(train['target'],oof_qda)

# print(f'AUC: {auc:.5}')
# sample_submission['target'] = pred_qda

# sample_submission.to_csv('sub_qda.csv',index=False)

%%time

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

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof[idx1[test_index3]] = clf.predict_proba(train3[test_index3,:])[:,1]

        preds[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits

       

    #if k%64==0: print(k)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('Pseudo Labeled QDA scores CV =',round(auc,5))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)



import matplotlib.pyplot as plt

plt.hist(preds,bins=100)

plt.title('Final Test.csv predictions')

plt.show()