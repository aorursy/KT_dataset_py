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
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

import xgboost as xgb
from sklearn.model_selection import train_test_split
sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
train.shape, test.shape
train.info()
train.head()
train_cols = [c for c in train.columns if c not in ["ID_code", "target"]]

y_train = train["target"]
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
train_32 = train.drop(['ID_code','target'], axis =1).astype('float32')
train_32.info()
train_32.head()
sample_submission.info()
sample_submission.head()

x.shape
y.shape
Y_train.shape
X_train.shape
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



X_train[:,0] = le.fit_transform(X_train[:,0])

X_test[:,0] = le.fit_transform(X_test[:,0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify = y, random_state =0)
X_train, X_test, y_train, y_test = train_test_split(resamp_x, resamp_y, test_size = 0.2,stratify = resamp_y, random_state =0)
clf_xgb = xgb.XGBClassifier(objective='binary:logistic')

clf_xgb.fit(X_train,y_train)
#cv_val = clf_xgb.predict_proba(X_test)[:,1]
y_pred = clf_xgb.predict(X_test)
accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))
dataset_dmatrix = xgb.DMatrix(data = X,label = y)

dataset_dmatrix
params = {"objective":"reg:logistic"}

params
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)
print(cv_results)
print(1-cv_results["test-rmse-mean"].tail(1))
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)
print(cv_results)
print(cv_results["test-auc-mean"].tail(1))
submission_rfc = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred

    })

submission_rfc.to_csv('submission_rfc.csv', index=False)
submission_log = pd.DataFrame({'ID_code':test.ID_code.values})

submission_log['target'] = y_pred

submission_log.to_csv('submission_logreg.csv', index=False)
X_train = train.iloc[:, train.columns != 'target'].values

Y_train = train.iloc[:, 1].values

X_test = test.values
train.target.value_counts() 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[:,0] = le.fit_transform(X_train[:,0])

X_test[:,0] = le.fit_transform(X_test[:,0])
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, Y_train)
y_preds = clf_knn.predict(X_test)

y_preds
submission_log = pd.DataFrame({'ID_code':test.ID_code.values})

submission_log['target'] = y_preds

submission_log.to_csv('knn_submission.csv', index=False)
from sklearn.linear_model import LogisticRegression
logreg_clf = LogisticRegression()
logreg_clf.fit(X_train,Y_train)
predictions = logreg_clf.predict(X_test)
submission_log = pd.DataFrame({'ID_code':test.ID_code.values})

submission_log['target'] = predictions

submission_log.to_csv('submission_logreg.csv', index=False)
from sklearn.svm import SVC

svclassifier = SVC()
svclassifier.fit(X_train, y_train)