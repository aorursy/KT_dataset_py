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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
test.head()
train.head()
train.info()

test.info()
X, y = train.iloc[:,2:], train.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify = y)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
logreg=LogisticRegression()

logreg.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix

y_pred = logreg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
logreg_pred = logreg.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

def performance(Y_test, logist_pred):

    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]

    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)

    print('AUC:')

    print(auc(fpr, tpr))
performance(y_test, logreg_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predit_svc = gnb.predict(X_test)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix



accuracy_score(y_test, y_predit_svc)
confusion_matrix(y_test,y_predit_svc)
print(classification_report(y_test, y_predit_svc))
import pandas as pd

sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
train_id = train['ID_code']

y_train = train['target']

X_train = train.drop(['ID_code', 'target'], axis=1, inplace = False)



test_id = test['ID_code']

X_test = test.drop('ID_code', axis=1, inplace = False)
import xgboost as xgb
model_xgb = xgb.XGBRegressor(n_estimators=5, max_depth=4, learning_rate=0.5) 

model_xgb.fit(X_train, y_train)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



train_pred = model_xgb.predict(X_train)

print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))
xgb_preds = model_xgb.predict(X_test)

solution = pd.DataFrame({"ID_code":test_id, "target":xgb_preds})

solution.to_csv("xgb_submission.csv", index = False)
