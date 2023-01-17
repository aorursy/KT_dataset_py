# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

data_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

data_ss = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
data_train.info()
data_test.info()
data_ss.info()
data_train.head()
data_test.head()
data_train.describe()
data_test.describe()
data_train_upd = data_train.drop(['ID_code', 'target'], axis = 1).astype('float32')

data_train_upd.info()
data_train.head(3)
data_train_upd.head(3)
X_train = data_train.iloc[:, 2:].values

Y_train = data_train.target.values

X_test = data_test.iloc[:, 1:].values
X_train.shape
Y_train.shape
X_test
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()

log_model.fit(X_train,Y_train)
preds = log_model.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(Y_train,preds))
submission_log = pd.DataFrame({'ID_code':data_test.ID_code.values})

submission_log['target'] = preds

submission_log.to_csv('submission_logreg.csv', index=False)
from sklearn.naive_bayes import GaussianNB
feats = [x for x in data_train.columns if 'var_' in x]
nv = GaussianNB()
nv.fit(data_train[feats], data_train['target'])
data_test['target'] = nv.predict_proba(data_test[feats])[:, 1]
data_test[['ID_code', 'target']].to_csv('submission_GaussianNV.csv', index=False)
import numpy as np 

import pandas as pd 

import gc

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix

import xgboost as xgb
data_traindf = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

data_testdf = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
data_traindf.shape, data_testdf.shape
data_traindf.head()
data_train_cols = [c for c in data_traindf.columns if c not in ["ID_code", "target"]]

y_train = data_traindf["target"]
y_train.value_counts()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
params = {'tree_method': 'hist',

 'objective': 'binary:logistic',

 'eval_metric': 'auc',

 'learning_rate': 0.0936165921314771,

 'max_depth': 2,

 'colsample_bytree': 0.3561271102144279,

 'subsample': 0.8246604621518232,

 'min_child_weight': 53,

 'gamma': 9.943467991283027,

 'silent': 1}
oof_preds = np.zeros(data_traindf.shape[0])

sub_preds = np.zeros(data_testdf.shape[0])



feature_importance_df = pd.DataFrame()



for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data_traindf, y_train)):

    

    trn_x, trn_y = data_traindf[data_train_cols].iloc[trn_idx], y_train.iloc[trn_idx]

    val_x, val_y = data_traindf[data_train_cols].iloc[val_idx], y_train.iloc[val_idx]

    

    dtrain = xgb.DMatrix(trn_x, trn_y, feature_names=trn_x.columns)

    dval = xgb.DMatrix(val_x, val_y, feature_names=val_x.columns)

    

    clf = xgb.train(params=params, dtrain=dtrain, num_boost_round=4000, evals=[(dtrain, "Train"), (dval, "Val")],

        verbose_eval= 100, early_stopping_rounds=50) 

    

    oof_preds[val_idx] = clf.predict(xgb.DMatrix(val_x))

    sub_preds += clf.predict(xgb.DMatrix(data_testdf[data_train_cols])) / folds.n_splits



    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    fold_importance_df["fold"] = n_fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    print('\nFold %1d AUC %.6f & std %.6f' %(n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx]), np.std([oof_preds[val_idx]])))

    print('Fold %1d Precision %.6f' %(n_fold + 1, precision_score(val_y, np.round(oof_preds[val_idx])) ))

    print('Fold %1d Recall %.6f' %(n_fold + 1, recall_score(val_y, np.round(oof_preds[val_idx]) )))

    print('Fold %1d F1 score %.6f' % (n_fold + 1,f1_score(val_y, np.round(oof_preds[val_idx]))))

    print('Fold %1d Kappa score %.6f\n' % (n_fold + 1,cohen_kappa_score(val_y, np.round(oof_preds[val_idx]))))

    gc.collect()



print('\nCV AUC score %.6f & std %.6f' % (roc_auc_score(y_train, oof_preds), np.std((oof_preds))))

print('CV Precision score %.6f' % (precision_score(y_train, np.round(oof_preds))))

print('CV Recall score %.6f' % (recall_score(y_train, np.round(oof_preds))))

print('CV F1 score %.6f' % (f1_score(y_train, np.round(oof_preds))))

print('CV Kappa score %.6f' % (cohen_kappa_score(y_train, np.round(oof_preds))))
print(confusion_matrix(y_train, np.round(oof_preds)))
fig, ax = plt.subplots(1,1,figsize=(10,12)) 

xgb.plot_importance(clf, max_num_features=20, ax=ax)  
fig, ax = plt.subplots(1,1,figsize=(10,12)) 

xgb.plot_importance(clf, max_num_features=20, ax=ax, importance_type="gain", xlabel="Gain")
feature_importance_df.groupby(["feature"])["fscore",].mean().sort_values("fscore", ascending=False)
data_testdf['target'] = sub_preds
data_testdf.head()
oof_roc = roc_auc_score(y_train, oof_preds)

oof_roc
ss = pd.DataFrame({"ID_code":data_testdf["ID_code"], "target":data_testdf["target"]})

ss.to_csv("sctp_xgboost.csv", index=None)

ss.head()
ss.describe().T
data_train.head()
data_train.describe()
data_test.head()
data_test.describe()
X_train = data_train.iloc[:, data_train.columns != 'target'].values

Y_train = data_train.iloc[:, 1].values

X_test = data_test.values
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()





X_train[:,0] = le.fit_transform(X_train[:,0])

X_test[:,0] = le.fit_transform(X_test[:,0])

knn = KNeighborsClassifier(11)
knn.fit(X_train, Y_train)