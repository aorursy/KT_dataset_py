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

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')

df.head()
df['native.country'].unique()
df.isnull().sum()
df.describe()
df['capital.gain'] = df['capital.gain'].replace(0,df['capital.gain'].mean())

df['capital.loss'] = df['capital.loss'].replace(0,df['capital.loss'].mean())
df.head()
#df['workclass']=df['workclass'].replace('?','undef')

df['occupation'] = df['occupation'].replace('?','occu')

df['native.country'] = df['native.country'].replace('?','country')
df["sex"] = df["sex"].map({"Male": 0, "Female":1})
g=sns.barplot(x='sex',y='income',data=df)

g=g.set_ylabel('income >50k')

plt.show()

# df.fillna(df.mean(),inplace=True)



df['workclass'].unique
df=df.drop(columns='workclass',axis=1)
from sklearn.preprocessing import StandardScaler , LabelEncoder ,OneHotEncoder

label = LabelEncoder()

df['marital.status'] = label.fit_transform(df['marital.status'])

df['race'] = label.fit_transform(df['race'])

df['sex'] = label.fit_transform(df['sex'])

df['education'] = label.fit_transform(df['education'])

#df['workclass'] = label.fit_transform(df['workclass'])

df['occupation'] = label.fit_transform(df['occupation'])

df['native.country'] = label.fit_transform(df['native.country'])

df['relationship'] = label.fit_transform(df['relationship'])

df['income'] = label.fit_transform(df['income'])
df['native.country'].unique()

df.columns
X = df.drop(columns='income')

y=df['income']

scaler = StandardScaler()

X_scaled=scaler.fit_transform(X)

fig,ax=plt.subplots(figsize=(25,15),facecolor='white')

sns.boxplot(data=df,ax=ax,width=0.5,fliersize=4)
q=df['fnlwgt'].quantile(0.80)

data_cleaned = df[df['fnlwgt']<q]

q=df['age'].quantile(0.99)

data_cleaned = df[df['age']<q]

q=df['education'].quantile(0.99)

data_cleaned = df[df['education']<q]

q=df['education.num'].quantile(0.99)

data_cleaned=df[df['education.num']<q]

q=df['capital.gain'].quantile(0.95)

data_cleaned=df[df['capital.gain']<q]

q=df['race'].quantile(0.99)

data_cleaned=df[df['race']<q]

q=df['capital.loss'].quantile(0.98)

data_cleaned=df[df['capital.loss']<q]

q=df['education.num'].quantile(0.99)

data_cleaned=df[df['education.num']<q]

q=df['native.country'].quantile(0.98)

data_cleaned=df[df['native.country']<q]

q=df['hours.per.week'].quantile(0.98)

data_cleaned=df[df['hours.per.week']<q]
df.plot()
sns.heatmap(df.corr())
y.unique
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=135)

rf= RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred_class = rf.predict(X_test)

accuracy_score(y_pred_class,y_test)
print(rf.get_params())
random_grid = {'n_estimators': [1,2,3,4,5,6,7,15,20,25,260,600,900],

               'max_features': ['auto', 'sqrt'],

               'max_depth': [10, 20, 30, 40, 50, 60],

               'min_samples_split': [2, 5, 10],

               'min_samples_leaf': [1, 2, 4],

               'bootstrap': [True, False]}
from sklearn.model_selection import RandomizedSearchCV

randomcv= RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
randomcv.fit(X_train,y_train)
randomcv.best_params_
model = RandomForestClassifier(n_estimators=260,min_samples_split=10,min_samples_leaf=4,max_features='auto',max_depth=40,bootstrap=True)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

accuracy_score(y_test,y_predict)