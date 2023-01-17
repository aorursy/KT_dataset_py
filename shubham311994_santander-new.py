# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import norm, rankdata

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from sklearn import preprocessing







from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc, roc_auc_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
t_d = test_data.copy()
test_data.drop(columns=['ID_code'],axis=1,inplace=True)
#data.describe().transpose()
data.info()
#X.columns
#Y.head()
# from sklearn.decomposition import PCA

# pca = PCA(n_components=110)

# X_dat_pca = pca.fit_transform(X)
#X_dat_pca[0]
#test_data_pca=pca.transform(test_data)
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X_dat_pca,Y.values,test_size=0.10,random_state=42)

one_indices= np.array(data[data.target==1].index)

zero_indices = np.array(data[data.target==0].index)

#now let us a define a function for make undersample data with different proportion

#different proportion means with different proportion of normal classes of data

def undersample(zero_indices,one_indices,times):#times denote the normal data = times*fraud data

    Normal_indices_undersample = np.array(np.random.choice(zero_indices,(times*Count_insincere_transacation),replace=False))

    print(len(Normal_indices_undersample))

    undersample_data= np.concatenate([one_indices,Normal_indices_undersample])



    undersample_data = data.iloc[undersample_data,:]

    #print(undersample_data)

    print(len(undersample_data))



#     print("the normal transacation proportion is :",len(undersample_data[undersample_data.target==0])/len(undersample_data))

#     print("the fraud transacation proportion is :",len(undersample_data[undersample_data.target==1])/len(undersample_data))

    print("total number of record in resampled data is:",len(undersample_data))

    return(undersample_data)

Count_insincere_transacation = len(data[data["target"]==1]) # fraud by 1
Undersample_data = undersample(zero_indices,one_indices,3)
df = Undersample_data.copy()

Y = df['target']

df.drop(columns=['target','ID_code'],axis=1,inplace=True)

X = df.loc[:,:]
from sklearn.decomposition import PCA

pca = PCA(n_components=110)

X = preprocessing.normalize(X)

X = preprocessing.scale(X)



X_dat_pca = pca.fit_transform(X)

X_dat_pca = preprocessing.normalize(X_dat_pca)

X_dat_pca = preprocessing.scale(X_dat_pca)
Y=Y.values
np.isnan(np.min(X_dat_pca))
test_data = preprocessing.normalize(test_data)

test_data = preprocessing.scale(test_data)



test_data_pca=pca.transform(test_data)

test_data_pca = preprocessing.normalize(test_data_pca)

test_data_pca = preprocessing.scale(test_data_pca)

X_dat_pca=X_dat_pca.astype(float)
test_data_pca=test_data_pca.astype(float)
NFOLDS = 30

RANDOM_STATE = 120





folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True,random_state=RANDOM_STATE)

oof_preds = np.zeros((len(X_dat_pca), 1))

test_preds = np.zeros((len(test_data_pca), 1))

roc_cv =[]
type(Y)
for fold_, (trn_, val_) in enumerate(folds.split(Y, Y)):

    print("Current Fold: {}".format(fold_))

    trn_x, trn_y = X_dat_pca[trn_, :], Y[trn_]

    val_x, val_y = X_dat_pca[val_, :], Y[val_]

    #print(trn_)

    #print(trn_y)

    

    clf = Pipeline([

        ('lr_clf', LogisticRegression(solver='lbfgs', max_iter=10000))

    ])



    clf.fit(trn_x, trn_y)



    val_pred = clf.predict_proba(val_x)[:,1]

    test_fold_pred = clf.predict_proba(test_data_pca)[:,1]

    

    roc_cv.append(roc_auc_score(val_y, val_pred))

    

    print("AUC = {}".format(roc_auc_score(val_y, val_pred)))

    oof_preds[val_, :] = val_pred.reshape((-1, 1))

    test_preds += test_fold_pred.reshape((-1, 1))
# param = {

#     'max_depth': 20,  # the maximum depth of each tree

#     'eta': 0.001,  # the training step for each iteration

#     'silent': 1,

#     'objective': 'binary:logistic'

#      } 

# param['nthread'] = 4

# # the number of classes that exist in this datset

# num_round = 1000  # the number of training iterations

# for fold_, (trn_, val_) in enumerate(folds.split(Y, Y)):

#     print("Current Fold: {}".format(fold_))

#     trn_x, trn_y = X_dat_pca[trn_, :], Y[trn_]

#     val_x, val_y = X_dat_pca[val_, :], Y[val_]

#     #print(trn_)

#     #print(trn_y)

#     dtrain = xgb.DMatrix(trn_x, label=trn_y)

#     dtest = xgb.DMatrix(val_x, label=val_y)

#     dtest1=xgb.DMatrix(test_data_pca)

    

#     bst = xgb.train(param, dtrain, num_round)



#     preds = bst.predict(dtest)

#     #preds=int(preds)

#     #test_fold_pred = bst.predict(dtest1)

#     #best_preds = np.asarray([np.argmax(line) for line in preds])

    

#     roc_cv.append(roc_auc_score(val_y, preds))

    

#     print("AUC = {}".format(roc_auc_score(val_y, preds)))

#     #oof_preds[val_, :] = preds.reshape((-1, 1))

#     #test_preds += test_fold_pred.reshape((-1, 1))
test_preds=test_preds/NFOLDS
# from sklearn.linear_model import LogisticRegression

# lgr = LogisticRegression(solver='lbfgs',max_iter=1000)
# lgr.fit(X_train,y_train)

# lgr_training_predictions = lgr.predict(X_train)

# print(lgr_training_predictions.shape)

# print(y_train.shape)

# from sklearn.metrics import accuracy_score

# print(accuracy_score(y_train,lgr_training_predictions))
# lgr_test_pred=lgr.predict(X_test)

# print(accuracy_score(y_test,lgr_test_pred))
# test_data_predictions = lgr.predict(test_data_pca)
# test_data_predictions[:20]
print("Saving submission file")

sample = pd.read_csv('../input/sample_submission.csv')

sample.target = test_preds.astype(float)

sample.ID_code = t_d["ID_code"]

sample.to_csv('submission.csv', index=False)
# submission_lgr = pd.DataFrame({

#         "ID_code": t_d["ID_code"],

#         "target": test_preds

#     })

# submission_lgr.to_csv('submission_ens.csv', index=False)
# from sklearn.decomposition import PCA

# pca = PCA(n_components=None)

# pca.fit(X)
# a = np.array(pca.explained_variance_ratio_)
# su=0.0

# for i,b in enumerate(a):

#     su = su + b

#     if(su<=0.95):

#         print(i)