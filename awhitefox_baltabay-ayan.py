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
dataset_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

dataset_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

dataset_train.head(10)
dataset_train.describe()
dataset_test.head(10)
dataset_test.describe()
dataset_train.info()
dataset_train_32 = dataset_train.drop(['ID_code','target'], axis=1).astype('float16')
dataset_train_32.info()
X_train = dataset_train_32.values

X_train
y_train = dataset_train.target.astype('uint8').values

y_train
X_test = dataset_test.iloc[:, 1:].astype('float16').values

X_test
X = X_train

y = y_train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y, random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
x_test = dataset_test.iloc[:,2:]
x_test2 = dataset_test.iloc[:,1:].values
y_pred = logreg.predict(x_test2)

#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
id_n = dataset_test.ID_code.values

submission_logreg = pd.DataFrame({

    "ID_code": dataset_test["ID_code"],

    "target": y_pred

})

submission_logreg.to_csv('submission_logreg.csv', index=False)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.classification_report(y_test, y_pred))
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split
bnb = BernoulliNB(binarize=0.0)
bnb.fit(X_train, y_train)
bnb.score(X_test, y_test)
y_pred = bnb.predict(x_test2)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Accuracy:",metrics.classification_report(y_test, y_pred))
submission_naive_bayes = pd.DataFrame({

    "ID_code": dataset_test["ID_code"],

    "target": y_pred

})

submission_naive_bayes.to_csv('submission_naive_bayes.csv', index=False)
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb
xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )
xg_cl.fit(X_train, y_train)
y_pred = xg_cl.predict(x_test2)
import numpy as np

accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))
param_grid = {'max_depth': [6,7,8], 'gamma': [1, 2, 4], 'learning_rate': [1, 0.1, 0.01], 'objective':['binary:logistic'], 'eval_metric': ['auc'],'tree_method': ['gpu_hist'],'n_gpus': [1]}
dataset_dmatrix = xgb.DMatrix(data = X,label = y)

dataset_dmatrix
params = {"objective":"reg:logistic", "max_depth":3}

params
# Perform cross-validation: cv_results

cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)
print(cv_results)
print(1-cv_results["test-rmse-mean"].tail(1))
# Perform cross_validation: cv_results

cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)# Perform cross_validation: cv_results

cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)
print(cv_results)
print(cv_results["test-auc-mean"].tail(1))
submission_xgboots = pd.DataFrame({

    "ID_code": dataset_test["ID_code"],

    "target": y_pred

})

submission_xgboots.to_csv('submission_xgboots.csv', index=False)