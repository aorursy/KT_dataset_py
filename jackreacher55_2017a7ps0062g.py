import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
cd /kaggle/input/eval-lab-1-f464-v2
df=pd.read_csv("train.csv")
df.head(10)
df.isnull().any().any()
df.isnull().sum()
df.describe()
df.info()
df1 = df[df.isna().any(axis=1)]
df1
df.head()
df.fillna(value=df.mean(),inplace=True)
df.isnull().any().any()
df.corr()
X_encoded = pd.get_dummies(df['type'])

X_encoded.head()
df['new'] = X_encoded['new']

df['old'] = X_encoded['old']
Selected_features = ['id','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','new','old']
#cols = df.columns.tolist()
#cols = cols[:-2] + cols[-1:] +cols[-2:-1] 
#df.columns = cols

#df.head()
df.drop(['type'], axis=1).head()
df.corr()
test = pd.read_csv('test.csv')

y = df['rating'].copy()

#x_test = test[Selected_features]
from sklearn import ensemble

from sklearn.metrics import mean_squared_error

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)

X_train = df[Selected_features]

y_train = df['rating']

clf.fit(X_train, y_train)
test.fillna(value=test.mean(),inplace=True)

X_encoded = pd.get_dummies(test['type'])

X_encoded.head()
test['new'] = X_encoded['new']

test['old'] = X_encoded['old']
test = test.drop(['type'], axis=1)

test.corr()
test.isnull().any().any()

selected = ['id','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','new','old']

rating = clf.predict(test[selected])
import sys

rating = np.round(rating)

np.set_printoptions(threshold=sys.maxsize)

rating.astype(int)
df_ans = pd.DataFrame()

test = pd.read_csv('test.csv')

df_ans['id'] = test['id']

df_ans['rating'] = rating
#df_ans.to_csv ('/kaggle/Output/2017A7PS0062G.csv')

!cd /kaggle
!mkdir /kaggle/Output
df_ans.to_csv ('/kaggle/Output/2017A7PS0062G.csv')