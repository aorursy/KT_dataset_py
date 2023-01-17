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

sns.set(color_codes = True)

%matplotlib inline
dataset = pd.read_csv('../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv',header = None)
dataset.head()
columns = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataset.columns = columns
dataset.head()
dataset.info()
dataset.describe().T
dataset[dataset['plas'] == 0].shape
dataset[dataset['pres'] == 0].shape
dataset[dataset['skin'] == 0].shape
dataset[dataset['test'] == 0].shape
dataset[dataset['mass'] == 0].shape
dataset[['plas','pres','skin','test','mass']].median()
dataset[['plas','pres','skin','test','mass']] = dataset[['plas','pres','skin','test','mass']].apply(lambda x: x.replace(0,x.median()))
dataset.head()
dataset.iloc[:,:-1] = dataset.iloc[:,:-1].astype('float64')
dataset.dtypes
dataset.describe().T
dataset.corr()['class'].plot(kind = 'bar')

plt.show()
plt.figure(figsize=(12,6))

sns.heatmap(dataset.corr(),annot = True)
dataset['class'].value_counts()
sns.pairplot(dataset,hue = 'class')
X = dataset.iloc[:,:-1]

y = dataset.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y,test_size= 0.3,random_state=42)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier
dtree = DecisionTreeClassifier(criterion = 'gini',max_depth = 5)

dtree.fit(X_train,y_train)

print('Training Score : ',dtree.score(X_train,y_train))

print('Testing Score : ',dtree.score(X_test,y_test))

y_pred = dtree.predict(X_test)

y_pred_prob = dtree.predict_proba(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print('<--------Confusion Matrix-------->\n',confusion_matrix(y_test,y_pred))

print('<--------Classification Report-------->\n',classification_report(y_test,y_pred))

rf = RandomForestClassifier(n_estimators=10,max_features=4, random_state=42)

rf.fit(X_train,y_train)

print('Training Score : ',rf.score(X_train,y_train))

print('Testing Score : ',rf.score(X_test,y_test))

y_pred = rf.predict(X_test)

y_pred_prob = rf.predict_proba(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print('<--------Confusion Matrix-------->\n',confusion_matrix(y_test,y_pred))

print('<--------Classification Report-------->\n',classification_report(y_test,y_pred))



dtree_1 = DecisionTreeClassifier()
dtree_1.fit(X_train,y_train)
abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),n_estimators=100)

abc.fit(X_train,y_train)

print('Training Score : ',abc.score(X_train,y_train))

print('Testing Score : ',abc.score(X_test,y_test))

y_pred = abc.predict(X_test)

y_pred_prob = abc.predict_proba(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print('<--------Confusion Matrix-------->\n',confusion_matrix(y_test,y_pred))

print('<--------Classification Report-------->\n',classification_report(y_test,y_pred))





gb = GradientBoostingClassifier(learning_rate=0.05,n_estimators=50,max_depth=3)

gb.fit(X_train,y_train)

print('Training Score : ',gb.score(X_train,y_train))

print('Testing Score : ',gb.score(X_test,y_test))

y_pred = gb.predict(X_test)

y_pred_prob = gb.predict_proba(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print('<--------Confusion Matrix-------->\n',confusion_matrix(y_test,y_pred))

print('<--------Classification Report-------->\n',classification_report(y_test,y_pred))





xgb = XGBClassifier()

xgb.fit(X_train,y_train)

print('Training Score : ',xgb.score(X_train,y_train))

print('Testing Score : ',xgb.score(X_test,y_test))

y_pred = xgb.predict(X_test)

y_pred_prob = xgb.predict_proba(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print('<--------Confusion Matrix-------->\n',confusion_matrix(y_test,y_pred))

print('<--------Classification Report-------->\n',classification_report(y_test,y_pred))





lgbm = LGBMClassifier(max_depth=3,

    learning_rate=0.1,

    n_estimators=50,)

lgbm.fit(X_train,y_train)

print('Training Score : ',lgbm.score(X_train,y_train))

print('Testing Score : ',lgbm.score(X_test,y_test))

y_pred = lgbm.predict(X_test)

y_pred_prob = lgbm.predict_proba(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print('<--------Confusion Matrix-------->\n',confusion_matrix(y_test,y_pred))

print('<--------Classification Report-------->\n',classification_report(y_test,y_pred))





cat = CatBoostClassifier(iterations=100,learning_rate=0.1)

cat.fit(X_train,y_train,plot = True)

print('Training Score : ',cat.score(X_train,y_train))

print('Testing Score : ',cat.score(X_test,y_test))

y_pred = cat.predict(X_test)

y_pred_prob = cat.predict_proba(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print('<--------Confusion Matrix-------->\n',confusion_matrix(y_test,y_pred))

print('<--------Classification Report-------->\n',classification_report(y_test,y_pred))





from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator=cat,X=X_train,y=y_train,scoring = 'accuracy',n_jobs = -1,verbose = 100,cv = 10)
cvs
cvs.mean()