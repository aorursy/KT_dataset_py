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
data=pd.read_csv('../input/predicting-earnings-from-census-data/census.csv')
data.info()
#what is the num of obsvn in the dataset

#31978
data.shape[0]

#num of rows
#num of columns

data.shape[1]
data.columns
#how is the dataset indexed?

print(data.index)
data.head()
data.age.mean()
data.age.median()
data.age.std()
data.isnull().sum()
data.over50k.unique()
def f(x):

    if x=='<=50K':

        return True

    if x=='>=50K':

        return False
#data.over50k=data.over50k.apply(f)
#data.over50k.value_counts()
data
a=data[['age']]

a.plot(kind='hist')
data['sex'].value_counts()
data['race'].value_counts()
data['maritalstatus'].value_counts()
data.hoursperweek.mean()
data.hoursperweek.median()
data.hoursperweek.max()
data.loc[data['hoursperweek']>=99]['race'].value_counts()
data.loc[data['hoursperweek']>=99]['sex'].value_counts()
data.loc[data['hoursperweek']>=99]['over50k'].value_counts(['percentage'])
data['hoursperweek'].plot(kind='hist')
data.groupby(by=['sex'])['age'].mean()

#avg age by gender

data.dtypes
data.education.nunique()
data.capitalloss.nunique()
data.hoursperweek.describe()
data.age.describe()
#what is the age with least occur

data.age.value_counts().tail()
data.hoursperweek.value_counts().tail()
#what is the educational qualification of the highest num of working hours individual

data.sort_values(by='hoursperweek',ascending=False).head()
data.education.value_counts()
data.iloc[:,0:7]
data.iloc[:,:-3]
data[['capitalloss','capitalgain']]
data.iloc[2:7,2:6]
data.iloc[4:,:]
data.iloc[:4,:]
data.iloc[:,2:7]
data[data.age>50]
#for each gender cat print the stat for hoursperweek and age

data.groupby('sex').age.describe()
data.groupby('sex').hoursperweek.describe()
data.groupby('workclass').age.mean()
data.workclass.value_counts()
def gen_to_num(x):

    if x=='Male':

        return 1

    if x=='Female':

        return 0
#data['sex']=data['sex'].apply(lambda x :gen_to_num(x))
data
d={'Male':1,'Female':0}
data.sex.value_counts()
data.sex.dtype
data.corr()
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
model_log=LogisticRegression()
label_encoder=preprocessing.LabelEncoder()
data['sex']= label_encoder.fit_transform(data['sex'])
data['workclass']= label_encoder.fit_transform(data['workclass'])

data['education']= label_encoder.fit_transform(data['education'])



data['occupation']= label_encoder.fit_transform(data['occupation'])

data['relationship']= label_encoder.fit_transform(data['relationship'])

data['race']= label_encoder.fit_transform(data['race'])

data['nativecountry']= label_encoder.fit_transform(data['nativecountry'])

data['over50k']= label_encoder.fit_transform(data['over50k'])

data['maritalstatus']= label_encoder.fit_transform(data['maritalstatus'])
data
X=data.drop('over50k',axis=1)
X
Y=data['over50k']
Y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.6, random_state=42)
model_log.fit(X_train,y_train)
model_log.coef_
yhat=model_log.predict(X_test)
yhat
model_log.score
model_log.score(X_train,y_train)
model_log.coef_
model_log.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier
decision_tree=DecisionTreeClassifier(random_state=0,max_depth=2)
decision_tree.fit(X_train,y_train)
import seaborn as sns; sns.set()

from matplotlib import pyplot as plt
yhat1=decision_tree.predict(X_test)
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

metrics.accuracy_score(y_test, yhat1)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
y1=rf.predict(X_test)
metrics.accuracy_score(y_test, y1)