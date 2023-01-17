%matplotlib inline

import matplotlib
import pandas as pd
adult = pd.read_csv('../input/base-adult-teste/train_data.csv',na_values='?')
adult.shape
adult.head()
adult.isnull().sum()
adult_full = adult.dropna()
adult_full.shape
import seaborn as sns 
adult_full['age_range'] = pd.cut(adult_full['age'],bins=[0,20,40,60,80,100])
sns.countplot(x='age_range', data = adult_full)
sns.countplot(x='age_range',hue='income', data = adult_full)
adult_full['age'].describe()
adult_full['workclass'].value_counts()
sns.countplot(x='workclass',hue='income', data = adult_full)
adult_full[adult_full['income']=="<=50K"].workclass.value_counts()
adult_full[adult_full['income']==">50K"].workclass.value_counts()
adult_full['fnlwgt'].describe()
adult_full['education'].value_counts()
adult_full[adult_full["income"]=="<=50K"].education.value_counts()
adult_full[adult_full["income"]==">50K"].education.value_counts()
adult_full['education.num'].value_counts()
sns.countplot(x='education.num',hue='income', data = adult_full)
sns.countplot(x='marital.status',hue='income', data = adult_full)
adult_full['marital.status'].value_counts()
adult_full[adult_full["income"]==">50K"]['marital.status'].value_counts()
adult_full['occupation'].value_counts()
adult_full[adult_full["income"]==">50K"]['occupation'].value_counts()
adult_full['relationship'].value_counts()
sns.countplot(x='relationship',hue='income', data = adult_full)
sns.countplot(x='race',hue='income', data = adult_full)
adult_full['race'].value_counts()
adult_full[adult_full["income"]==">50K"]['race'].value_counts()
sns.countplot(x='sex', data = adult_full)
sns.countplot(x='sex',hue='income', data = adult_full)
adult_full["capital.gain"].value_counts()
adult_full['cg'] = 1
adult_full.loc[adult_full['capital.gain']==0,'cg']=0
adult_full.loc[adult_full['capital.gain']!=0,'cg']=1
adult_full.head()
adult_full["cg"].value_counts()
sns.countplot(x='cg',hue='income', data = adult_full)
adult_full["capital.loss"].value_counts()
adult_full['cl'] = 1
adult_full.loc[adult_full['capital.loss']==0,'cl']=0
adult_full.loc[adult_full['capital.loss']!=0,'cl']=1
adult_full["cl"].value_counts()
sns.countplot(x='cl',hue='income', data = adult_full)
adult_full[adult_full["income"]==">50K"]["hours.per.week"].describe()
adult_full[adult_full["income"]=="<=50K"]["hours.per.week"].describe()
adult_full['native.country'].value_counts()
adult_full[adult_full["income"]==">50K"]["native.country"].value_counts()
adult_full[adult_full["income"]=="<=50K"]["native.country"].value_counts()
Xadult = adult_full[["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]]
Yadult=adult_full.income

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(knn,Xadult,Yadult,cv=10)
scores1
scores1.mean()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn,Xadult,Yadult,cv=10)
scores
scores.mean()
from sklearn import preprocessing
num_adult_full = adult_full.apply(preprocessing.LabelEncoder().fit_transform)
num_adult_full.head()

Xadult2 = num_adult_full[["age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week","native.country"]]
Yadult=adult_full.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn,Xadult2,Yadult,cv=10)
scores

scores.mean()
Xadult2 = num_adult_full[["age","workclass","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week","native.country"]]
Yadult=adult_full.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn,Xadult2,Yadult,cv=10)
scores

scores.mean()
Xadult3 = num_adult_full[["age","workclass","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week"]]
Yadult=adult_full.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn,Xadult3,Yadult,cv=10)
scores

scores.mean()
Xadult4 = num_adult_full[["age","workclass","education.num","marital.status","relationship","race","sex","capital.gain","capital.loss","hours.per.week"]]
Yadult=adult_full.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn,Xadult4,Yadult,cv=10)
scores

scores.mean()