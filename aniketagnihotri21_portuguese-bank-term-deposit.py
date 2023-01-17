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

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

import os
fpath = '/kaggle/input/portuguese-bank-marketing-data-set/bank-full.csv'

bankdf = pd.read_csv(fpath, sep = ';')

bankdf.head()
bankdf.columns
bankdf.shape
bankdf.info()
bankdf['job'] = bankdf['job'].astype({'job':'category'})

bankdf['marital'] = bankdf['marital'].astype({'marital':'category'})

bankdf['education'] = bankdf['education'].astype({'education':'category'})

bankdf['default'] = bankdf['default'].astype({'default':'category'})

bankdf['housing'] = bankdf['housing'].astype({'housing':'category'})

bankdf['loan'] = bankdf['loan'].astype({'loan':'category'})

bankdf['contact'] = bankdf['contact'].astype({'contact':'category'})

bankdf['month'] = bankdf['month'].astype({'month':'category'})

bankdf['poutcome'] = bankdf['poutcome'].astype({'poutcome':'category'})

bankdf['y'] = bankdf['y'].astype({'y':'category'})
# Running Info again

bankdf.info()
bankdf.isnull().sum()
# Five Point Summary

bankdf.describe()
bankdf['y'].value_counts(normalize=True)*100
sns.boxenplot(bankdf['age'])
sns.boxplot(bankdf['balance'])
sns.boxplot(bankdf['day'])
sns.boxplot(bankdf['duration'])
sns.boxplot(bankdf['campaign'])
sns.distplot(bankdf['age'], bins=10, kde=False)
sns.distplot(bankdf['balance'],bins=20, kde=False)
bankdf['job']= bankdf['job'].replace({'entrepreneur':1, 'management':2,'technician':3,'admin.':4,'services':5, 'self-employed':6,'blue-collar':7,'retired':8,'unemployed':9,'housemaid':10,'student':11,'unknown':12})

bankdf['education'] = bankdf['education'].replace({'primary':1,'secondary':2,'tertiary':3,'unknown':4})

bankdf['housing'] = bankdf['housing'].replace({'yes':1, 'no':0})

bankdf['default'] = bankdf['default'].replace({'yes':1, 'no':0})

bankdf['loan'] = bankdf['loan'].replace({'yes':1, 'no':0})

bankdf['y'] = bankdf['y'].replace({'yes':1, 'no':0})
sns.barplot(x = 'y', y = 'age', data = bankdf)
sns.barplot(x = 'y', y='balance', data=bankdf)
sns.barplot(x='y', y='day', data=bankdf)
sns.boxenplot(x='y', y=('duration'), data=bankdf)
sns.boxenplot(x = 'y', y ='campaign', data=bankdf )
sns.boxenplot(x = 'y', y ='pdays', data=bankdf )
sns.boxenplot(x = 'y', y ='previous', data=bankdf )
bankdf = bankdf.drop(columns=['marital', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])
#creting X and y variable 

X = bankdf.drop('y', axis=1)

y = bankdf['y']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state =1)
log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)
log_reg.score(X_train,y_train)
log_reg.fit(X_test,y_test)
log_reg.score(X_test,y_test)
log_reg = log_reg.predict(X_test)
print(classification_report(log_reg,y_test))
knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

knn.score(X_train,y_train)
knn.fit(X_test,y_test)
knn.score(X_test,y_test)
knn_pred = knn.predict(X_test)
print(classification_report(knn_pred,y_test))
dtree_model = DecisionTreeClassifier()

dtree_model.fit(X_train, y_train)
Dtree_score_trn= dtree_model.score(X_train, y_train)

Dtree_score_trn
dtree_model.fit(X_test,y_test)
Dtree_score_tst = dtree_model.score(X_test,y_test)

Dtree_score_tst
y_pred = dtree_model.predict(X_test)
print(classification_report(y_pred,y_test))
RF_model = RandomForestClassifier()

RF_model.fit(X_train, y_train)
RF_score_trn = RF_model.score(X_train, y_train)

RF_score_trn
RF_score_tst = RF_model.score(X_test, y_test)

RF_score_tst
predictions_RF = RF_model.predict(X_test) 
print (classification_report(y_test, predictions_RF))
rfclass = RandomForestClassifier(n_estimators = 50)

rfclass.fit(X_train, y_train)
rfclass.fit(X_test, y_test)
RF_score_trn = rfclass.score(X_train, y_train)

RF_score_trn
RF_score_tst = rfclass.score(X_test, y_test)

RF_score_tst
rf_pred = rfclass.predict(X_test)
print(classification_report(y_test,rf_pred))
adb = AdaBoostClassifier(n_estimators= 50, learning_rate=1.0, random_state=22)

adb.fit(X_train,y_train)
AD_score_trn = adb.score(X_train,y_train)

AD_score_trn
adb.fit(X_test,y_test)
AD_score_tst = adb.score(X_test,y_test)

AD_score_tst
adb_pred = adb.predict(X_test)
print(classification_report(y_test,adb_pred))
bagg_cl = BaggingClassifier(n_estimators= 50)

bagg_cl.fit(X_train,y_train)
Bagg_score_trn = bagg_cl.score(X_train,y_train)

Bagg_score_trn
bagg_cl.fit(X_test,y_test)
Bagg_score_tst = bagg_cl.score(X_test,y_test)

Bagg_score_tst
bagg_pred = bagg_cl.predict(X_test)
print(classification_report(bagg_pred,y_test))
d = {'Model': ['DTree', 'RF', 'AdaBoost', 'Bagging'], 

     'Training val':[Dtree_score_trn, RF_score_trn, AD_score_trn, Bagg_score_trn], 

     'Test Val': [Dtree_score_tst, RF_score_tst, AD_score_tst, Bagg_score_tst],}

print (d)
m_eval = pd.DataFrame(d)

m_eval
plt.figure(figsize = (10,7))

sns.set_style("darkgrid")

plt.plot(m_eval['Model'], m_eval['Training val'], marker = '*')

plt.plot(m_eval['Model'], m_eval['Test Val'], marker = 'x')

plt.show()