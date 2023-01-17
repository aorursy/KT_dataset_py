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
tests = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

trains = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

subms = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
import matplotlib.pyplot as plt

import seaborn as sns
trains
plt.figure(figsize=(7, 5))

plt.style.use('ggplot')

plt.hist(trains.target)

plt.show()
desc = trains.describe()

desc
means = list(desc[desc.target == 0.100490].values)[0][1:]

means = list(means)

means.index(max(means))
desc.var_120
max_mean_col = trains.var_120

plt.figure(figsize=(7, 5))

plt.hist(max_mean_col)

plt.show()
plt.figure(figsize=(7, 5))

plt.hist(max_mean_col, cumulative=True, density=True, bins=25)

plt.show()
from scipy.stats import norm
plt.plot(max_mean_col, norm.pdf(max_mean_col))

plt.show()
means.index(min(means))
min_mean_col = trains.var_120

plt.figure(figsize=(7, 5))

plt.hist(min_mean_col)

plt.show()
plt.figure(figsize=(7, 5))

plt.hist(min_mean_col, cumulative=True, density=True, bins=25)

plt.show()
plt.plot(min_mean_col, norm.pdf(min_mean_col))

plt.show()
np.corrcoef(max_mean_col, min_mean_col)
plt.plot(max_mean_col, min_mean_col)

plt.show()
x = trains.drop(columns='var_120')

x = x.drop(columns=['target', 'ID_code'])
x
y = trains.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
X_test
tests
log_reg=LogisticRegression()

log_reg.fit(X_train,Y_train)

y_pred=log_reg.predict(X_test)
import sklearn

sklearn.metrics.accuracy_score(Y_test,y_pred)
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y,log_reg.predict(x))
print(classification_report(y,log_reg.predict(x)))
import matplotlib.pyplot as plt

import seaborn as sn
cm=confusion_matrix(Y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test,y_pred)
pred = log_reg.predict(tests.drop(columns=['ID_code','var_120']))
submit_log_reg = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})

submit_log_reg.to_csv('submit_log_reg.csv', index=False)
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
metrics.accuracy_score(y,gnb.predict(x))
confusion_matrix(y,gnb.predict(x))
print(classification_report(y,gnb.predict(x)))
roc_auc_score(Y_test, y_pred)
pred = gnb.predict(tests.drop(columns=['ID_code','var_120']))
submit_nb = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})

submit_nb.to_csv('submit_nb.csv', index=False)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
dt_class = DecisionTreeClassifier(criterion = "entropy", max_depth=5, min_samples_split = 10 )
dt_class.fit(X_train, Y_train)
print(classification_report(y,dt_class.predict(x)))
pred = dt_class.predict(tests.drop(columns=['ID_code','var_120']))
submit_dt = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})

submit_dt.to_csv('submit_dt.csv', index=False)
roc_auc_score(y, dt_class.predict(x))
rf_class = RandomForestClassifier()
rf_class.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score
prediction = rf_class.predict(X_test)

accuracy_score(Y_test,prediction)
print(classification_report(y,rf_class.predict(x)))
pred = rf_class.predict(tests.drop(columns=['ID_code','var_120']))
submit_rf = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})

submit_rf.to_csv('submit_rf.csv', index=False)
roc_auc_score(Y_test, rf_class.predict(y))
import xgboost as xx
x_cls = xx.XGBClassifier()
x_cls = x_cls.fit(X_train, Y_train)
y_pred = x_cls.predict(X_test)
pred = x_cls.predict(tests.drop(columns=['ID_code','var_120']))
submit_xg = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})

submit_xg.to_csv('submit_xg.csv', index=False)
roc_auc_score(Y_test, y_pred)