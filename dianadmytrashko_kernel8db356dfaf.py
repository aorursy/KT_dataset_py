# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/OnlineNewsPopularityReduced.csv")

df.head()

df
y=df['shares']

y
df=df.drop('url', axis=1)

df

df=df.drop('shares', axis=1)

df
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(df)

df_scaled = scaler.transform(df)
x=df_scaled

x
from sklearn.model_selection import train_test_split
df_train, df_test, y_train, y_test = train_test_split(

        df, y, test_size=0.25, random_state=19)
df_test.head()
from sklearn.neighbors import KNeighborsRegressor
knn1 = KNeighborsRegressor(n_neighbors=1)



knn1.fit(df_train, y_train)

knn2 = KNeighborsRegressor(n_neighbors=10)

knn2.fit(df_train, y_train)
y_pred1=knn1.predict(df_test)

y_pred1
y_pred2=knn2.predict(df_test)

y_pred2
from sklearn.metrics import mean_squared_error



m1=mean_squared_error(y_test, y_pred1)

m1
m2=mean_squared_error(y_test, y_pred2)

m2
m=abs(m2-m1)/m1

m
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsRegressor(n_neighbors=5)

scores = cross_val_score(knn, x, y, 

                         cv=kf, scoring='neg_mean_squared_error')

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 21)} # число соседей -- от 1 до 20

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='neg_mean_squared_error',

                        cv=kf) # или cv=kf

knn_grid.fit(df_train, y_train)
knn_grid.best_estimator_
knn_grid.best_score_
knn_grid.best_params_
pd.DataFrame(knn_grid.cv_results_).T
y_pred_b = knn_grid.predict(df_test)

mean_squared_error(y_test, y_pred_b)
best_knn = KNeighborsRegressor(n_neighbors=19)

y_pred_best = best_knn.fit(df_train, y_train).predict(df_test)

mean_squared_error(y_test, y_pred_best)
#3
#knn_params_p = {'p': np.linspace(1, 10, num=200)}

#knn_grid_p = GridSearchCV(knn, knn_params_p,scoring='neg_mean_squared_error',cv=kf) 

#knn_grid_p.fit(df_train, y_train)

#y_pred_p=knn_grid_p.predict(df_test)

#print (y_pred_p)

#scores_p = cross_val_score(knn_grid_p, x, y, cv=5, scoring='neg_mean_squared_error')

#scores_p
#mean_score_p=scores_p.mean()

#mean_score_p
knn_params_n = {'n_neighbors': np.arange(1, 51)} # число соседей -- от 1 до 50

knn_grid_n = GridSearchCV(knn, 

                        knn_params_n, 

                        scoring='neg_mean_squared_error',

                        cv=kf) # или cv=kf

knn_grid_n.fit(df_train, y_train)
knn_grid_n.best_params_
knn_grid_n.best_score_
df_groid = pd.DataFrame(knn_grid_n.cv_results_)

df_groid
from matplotlib.pyplot import plot
df_groid.plot(x='param_n_neighbors', y='mean_test_score');