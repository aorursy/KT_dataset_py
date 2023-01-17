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
import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style='darkgrid')

import matplotlib.pyplot as plt

from matplotlib import style

#sta matplotlib to inline and displays graphs below the corresponding cell.

%matplotlib inline

import os

from sklearn.datasets import *

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/purchase_prediction.csv")

df.head()
#lets drop the 'user id' which is not useful for our prediction further

df.drop("User ID",axis=1,inplace=True)
df.head()
df.shape
#Data Imputation

df.isnull().sum()
df.describe(include='all').T
df.info()
df.Purchased.value_counts()
#DATA WRANGLING

from sklearn import preprocessing  

df=df.apply(preprocessing.LabelEncoder().fit_transform)

df.head()
from statsmodels.stats.proportion import proportions_ztest
# for Gender and Attrition

pd.crosstab(df['Purchased'],df['Gender'])
count=np.array([77,66])

obs=np.array([204,196])
zstat,pvalue=proportions_ztest(count,obs)

print('z value: %0.3f, p value: %0.3f' %(zstat,pvalue))
from scipy.stats import ttest_ind
grp=df.groupby('Purchased')

grp_0=grp.get_group(0)

grp_1=grp.get_group(1)
mean1=grp_1.Age.mean()

mean1
mean0=grp_0.Age.mean()

mean0
ttest_ind(grp_0['Age'],grp_1['Age'])
#lets find  the pvalue for EstimatedSalary

grp=df.groupby('Purchased')

grp_0=grp.get_group(0)

grp_1=grp.get_group(1)
mean1=grp_1.EstimatedSalary.mean()

mean1
mean0=grp_0.EstimatedSalary.mean()

mean0
ttest_ind(grp_0['EstimatedSalary'],grp_1['EstimatedSalary'])
df.drop('Gender',1,inplace=True)
df.head()
y=df.Purchased

X=df.drop('Purchased',1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures,StandardScaler

se=StandardScaler()

X_train=se.fit_transform(X_train)

X_test=se.transform(X_test)
print(X_train.shape,X_test.shape)
#apply SMOTE viz resampling technique

from  imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=1,ratio=1.0)

X_train,y_train=sm.fit_sample(X_train,y_train)
## Apply Logistic Regression with balanced data by SMOTE

from sklearn.linear_model import LogisticRegression

smote=LogisticRegression()

smote.fit(X_train,y_train)

somote_pred=smote.predict(X_test)

#checking Accuracy

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

from sklearn.metrics import classification_report

from mlxtend.evaluate import confusion_matrix

accuracy_score(y_test,somote_pred)
print('classification:\n',classification_report(y_test,somote_pred))

f1_score(y_test,somote_pred)
print('recall score:',recall_score(y_test,somote_pred))
print('precision_score',precision_score(y_test,somote_pred))
sns.heatmap(confusion_matrix(y_test,somote_pred),annot=True)

plt.xlabel('Actual')

plt.ylabel('Predicted')
df.head()
y=df.Purchased

X=df.drop('Purchased',1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures,StandardScaler

se=StandardScaler()

X_train=se.fit_transform(X_train)

X_test=se.transform(X_test)
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
param={'n_neighbors':np.arange(1,50),'weights':['uniform','distance']}

gs=GridSearchCV(knn,param,cv=5,scoring='roc_auc')

gs.fit(X_train,y_train)
param={'n_neighbors':np.arange(1,50),'weights':['uniform','distance']}

gs=GridSearchCV(knn,param,cv=5,scoring='roc_auc')

gs.fit(X_train,y_train)
gs.best_params_
knn=KNeighborsClassifier(n_neighbors=23,weights='uniform')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print('Accuracy Score:',accuracy_score(y_test,y_pred))
print('the confusion matrix:\n',confusion_matrix(y_test,y_pred))
print('The classification report:',classification_report(y_test,y_pred))
print('recall score:',recall_score(y_test,y_pred))
print('precision_score',precision_score(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

plt.xlabel('Actual')

plt.ylabel('Predicted')
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
parms= {'criterion':['entropy','gini']}

grids=GridSearchCV(dt,parms,cv=10,scoring='roc_auc')

grids.fit(X_train,y_train)
grids.best_params_
dt=DecisionTreeClassifier(criterion='entropy',random_state=0)

dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
print('Accuracy Score:',accuracy_score(y_test,y_pred))
print('the confusion matrix:\n',confusion_matrix(y_test,y_pred))
print('The classification report:',classification_report(y_test,y_pred))
print('recall score:',recall_score(y_test,y_pred))
print('precision_score',precision_score(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

plt.xlabel('Actual')

plt.ylabel('Predicted')
#import a Library

from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(X_train,y_train)

y_pred=gb.predict(X_test)
print('Accuracy Score:',accuracy_score(y_test,y_pred))
print('the confusion matrix:\n',confusion_matrix(y_test,y_pred))
print('The classification report:',classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

plt.xlabel('Actual')

plt.ylabel('Predicted')
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

rf=RandomForestClassifier(n_estimators=50,random_state=0)
rf_var=[]

for val in np.arange(1,50):

    rf=RandomForestClassifier(criterion='entropy',n_estimators=val,random_state=0)

    kfold = KFold(shuffle=True,n_splits=5, random_state=0)

    cv_results = cross_val_score(rf, X, y, cv=kfold, scoring='roc_auc')

    rf_var.append(np.var(cv_results,ddof=1))

    print(val,np.var(cv_results,ddof=1))
x_axis=np.arange(1,50)

plt.plot(x_axis,rf_var)
rf=RandomForestClassifier(n_estimators=17,random_state=0)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print('Accuracy Score:',accuracy_score(y_test,y_pred))
print('the confusion matrix:\n',confusion_matrix(y_test,y_pred))
print('The classification report:',classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

plt.xlabel('Actual')

plt.ylabel('Predicted')
models=[]

models.append(('Logistic',smote))

models.append(('Naive',gb))

models.append(('knn',knn))

models.append(('DT',dt))

models.append(('RF',rf))
# evaluate each model in turn

results = []

names = []

scoring = 'roc_auc'

for name,model in models:

    kfold = KFold(n_splits=5, random_state=0,shuffle=True)

    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.var(ddof=1))

    print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()