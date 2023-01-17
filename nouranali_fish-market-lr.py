import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

%matplotlib inline

sns.set_style('whitegrid')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df= pd.read_csv('/kaggle/input/fish-market/Fish.csv')

df.head()
df.tail()
df.describe()
df.shape
df.Species.value_counts()
sns.catplot(x='Species' ,data=df,kind='count')
df.groupby('Species').median().sort_values('Weight',ascending=False)
g=sns.pairplot(df,hue='Species')

plt.show()
dfs=df[['Weight','Length1','Height','Width']]

std = StandardScaler()

dfs=pd.DataFrame(std.fit_transform(dfs),columns=dfs.columns)
X_train, X_test, y_train, y_test = train_test_split(dfs.drop('Weight',axis=1), dfs['Weight'], test_size=0.3, random_state=42)
lr=LinearRegression()

lr.fit(X_train,y_train)
print(lr.score(X_test,y_test) * 100)
yhat=lr.predict(X_test)

print(metrics.mean_squared_error(yhat,y_test))
rf=RandomForestRegressor()

rf.fit(X_train,y_train)
print(rf.score(X_test,y_test) * 100)
yhatrf=rf.predict(X_test)

print(metrics.mean_squared_error(yhatrf,y_test))
model_xgb = xgb.XGBRegressor(n_estimators=2000, max_depth=4, learning_rate=0.1, 

                             verbosity=1, silent=None, objective='reg:linear', booster='gbtree', 

                             n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 

                             subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0.2, reg_lambda=1.2, 

                             scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain')
model_xgb.fit(X_train, y_train)
yhatxgb= model_xgb.predict(X_test)
print(model_xgb.score(X_test,y_test) * 100)
print(metrics.mean_squared_error(yhatxgb,y_test))