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

import seaborn as sns



import pandas as pd

import numpy as np

from scipy.stats import skew

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

import xgboost as xgb



import warnings

warnings.simplefilter('ignore')



from scipy import stats

from scipy.stats import norm, skew #for some statistics



import os



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
import pandas as pd

sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
test.head()
train.head()
sns.countplot(x='target', data=train)
sns.distplot(train.var_0)

sns.distplot(train.var_10) 

sns.distplot(train.var_20) 
fig, ax = plt.subplots(ncols=2, figsize=(20, 4))



sns.distplot(train['var_0'], ax=ax[0], color='orange')

sns.distplot(train['var_1'], ax=ax[1], color='blue')



plt.show()
print(train.shape, test.shape)
print (train.isna().sum())

print (train.isnull().sum())

X, y = train.iloc[:,2:], train.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)
logreg=LogisticRegression()

logreg.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix

y_pred = logreg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([-0.2, 1.2])

plt.ylim([-0.2, 1.5])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



train_pred = logreg.predict(X_train)

print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))
from sklearn.naive_bayes import GaussianNB
gaus = GaussianNB()

gaus.fit(X_train,y_train)
y_pred = gaus.predict(X_test)
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred)
train_id = train['ID_code']

y_train = train['target']

X_train = train.drop(['ID_code', 'target'], axis=1, inplace = False)



test_id = test['ID_code']

X_test = test.drop('ID_code', axis=1, inplace = False)
model_xgb = xgb.XGBRegressor(n_estimators=5, max_depth=4, learning_rate=0.5) 

model_xgb.fit(X_train, y_train)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



train_pred = model_xgb.predict(X_train)

print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))
np.random.seed(11)
y_train = train['target']

x_train = train.drop(columns=['ID_code', 'target'])



y_train = y_train.astype('int8')

x_train = x_train.astype('float16')
from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from imblearn.ensemble import *

clf = BalancedRandomForestClassifier(n_estimators=500, 

                                     criterion='entropy', 

                                     n_jobs=-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train)
print('Score:', clf.score(x_train, y_train))
np.random.seed(11)
y_train = train['target']

x_train = train.drop(columns=['ID_code', 'target'])



y_train = y_train.astype('int8')

x_train = x_train.astype('float16')
from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from imblearn.ensemble import *

clf = BalancedRandomForestClassifier(n_estimators=500, 

                                     criterion='entropy', 

                                     n_jobs=-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train)
print('Score:', clf.score(x_train, y_train))
np.random.seed(11)
y_train = train['target']

x_train = train.drop(columns=['ID_code', 'target'])



y_train = y_train.astype('int8')

x_train = x_train.astype('float16')
from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from imblearn.ensemble import *

clf = BalancedRandomForestClassifier(n_estimators=500, 

                                     criterion='entropy', 

                                     n_jobs=-1)
clf.fit(x_train, y_train)