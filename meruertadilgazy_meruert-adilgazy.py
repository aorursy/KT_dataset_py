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

test = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/train.csv")
test.head()
train.head()
test.info()
train.info()
train.isnull().sum()
test.isnull().sum()
train.nunique()
test.nunique()
train['target'].unique()
train['target'].value_counts(normalize=True)
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(10, 7))

sns.countplot(train['target'])
train.shape , test.shape
train.describe()
pd.DataFrame.drop(train,columns=['ID_code'],axis=1,inplace=True)
X = train.iloc[:,1:]

y = train.iloc[:,0]
from sklearn.model_selection import train_test_split
X.shape, y.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size = 0.4, random_state = 123, stratify = y)
print('X_train shape is :', X_train.shape, '\ny_train shape is ',y_train.shape)
plt.figure(figsize=(10, 7))

sns.heatmap(train.corr())
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()

reg.fit(X,y)
y_pred = reg.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print('accuracy is: ',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm
import numpy as np
cm_sum = np.sum(cm, axis=1, keepdims=True)

cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)

nrows, ncols = cm.shape

for i in range(nrows):

    for j in range(ncols):

        c = cm[i, j]

        p = cm_perc[i, j]

        if i == j:

            s = cm_sum[i]

            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

        elif c == 0:

            annot[i, j] = ''

        else:

            annot[i, j] = '%.1f%%\n%d' % (p, c)



cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



fig, ax = plt.subplots(figsize=[5,2])



sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)
from sklearn.naive_bayes import GaussianNB
gaus = GaussianNB()

gaus.fit(X_train,y_train)
y_pred = gaus.predict(X_test)
print('accuracy is: ',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
from sklearn.naive_bayes import BernoulliNB
b = BernoulliNB()

b.fit(X_train,y_train)
y_pred = b.predict(X_test)
print('accuracy is: ',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)
from sklearn.utils.testing import ignore_warnings

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
xgb_cls = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=1,

              learning_rate=0.1, max_delta_step=0, max_depth=8,

              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)
param_grid = {'max_depth': [5,6,7,8], 'gamma': [1, 2, 4], 'learning_rate': [1, 0.1, 0.01, 0.001]}
test
xx_test = test[test.columns[1:]]
yy_test = test[test.columns[:1]]
yy_predd = xgb.predict(xx_test.values)
my_submission = pd.DataFrame({'ID_code': yy_test.ID_code,'target': yy_predd})

my_submission.to_csv('submission.csv', index=False)