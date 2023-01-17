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

from scipy.stats import skew

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from scipy import stats

from sklearn.metrics import confusion_matrix

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv', delimiter=',')

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

train.head()

data.dropna(inplace=True)

train.dropna(inplace=True)

test.dropna(inplace=True)
test.describe()

train.describe()

sns.countplot(x='target', data=train)
sns.distplot(train.var_0) 

sns.distplot(train.var_1) 

sns.distplot(train.var_2) 

sns.distplot(train.var_3) 
X, y = train.iloc[:,2:], train.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)
logreg=LogisticRegression()

logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 0.75])
from sklearn.svm import SVC
#vc = SVC(kernel='linear')

#svc.fit(X_train,y_train)
#svc.predict([X_test])
#GaussianNB().fit(X_train,y_train)
#y_gnb = GaussianNB().predict(X_test)
#classification_report(y_test, y_gnb)
#xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                #max_depth = 5, alpha = 10, n_estimators = 10)

#xgb.fit(X_train, y_train)
#y_xgb = model_xgb.predict(X_test)
#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

#predictions = gbm.predict(test_X)
#Xtree = DecisionTreeClassifier(random_state=0)

#Xtree.fit(X, y)
#y_tree = Xtree.predict(X_test)
#classification_report(y_test, y_tree)

#confusion_matrix(y_test,y_tree)
roc_auc_score(y_test, y_pred)