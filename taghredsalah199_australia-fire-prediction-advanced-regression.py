import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_fire= pd.read_csv('../input/fires-from-space-australia-and-new-zeland/fire_archive_M6_96619.csv')

df_fire
df_fire.info()
df_fire=df_fire.drop(['acq_date','acq_time','satellite','instrument','version','type'],axis=1)
daynight = pd.get_dummies(df_fire['daynight'],drop_first=True)

df_fire.drop(['daynight'],axis=1,inplace=True)

df_fire = pd.concat([df_fire,daynight],axis=1)

plt.figure(figsize=(20,10))

sns.heatmap( df_fire.isnull() , yticklabels=False ,cbar=False )
figure= plt.figure(figsize=(10,10))

sns.heatmap(df_fire.corr(), annot=True)

sns.pairplot(df_fire)
X= df_fire.drop('frp',axis=1)

y=df_fire['frp']
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X ,y , test_size=0.4, random_state=108)
from sklearn import metrics

from sklearn.model_selection import cross_val_score



results_df = pd.DataFrame()

columns = ["Model", "Cross Val Score", "MAE", "MSE", "RMSE", "R2"]



def evaluate(true, predicted):

    mae = metrics.mean_absolute_error(true, predicted)

    mse = metrics.mean_squared_error(true, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))

    r2_square = metrics.r2_score(true, predicted)

    return mae, mse, rmse, r2_square



def append_results(model_name, model, results_df, y_test, pred):

    results_append_df = pd.DataFrame(data=[[model_name, *evaluate(y_test, pred) , cross_val_score(model, X, y, cv=10).mean()]], columns=columns)

    results_df = results_df.append(results_append_df, ignore_index = True)

    return results_df
from sklearn.linear_model import RANSACRegressor

ransacReg= RANSACRegressor()

ransacReg.fit(X_train,y_train)

pred= ransacReg.predict(X_test)

results_df= append_results("Robust Regression",RANSACRegressor(),results_df,y_test,pred)

results_df
figure= plt.figure(figsize=(10,10))

sns.distplot((y_test,pred))

#To see the distribution between predection and acual value, if it normally distributed it means that model is correct
from sklearn.linear_model import Ridge

RidgeReg= Ridge()

RidgeReg.fit(X_train,y_train)

pred= RidgeReg.predict(X_test)

results_df= append_results("Ridge Regression",Ridge(),results_df,y_test,pred)

results_df
figure= plt.figure(figsize=(10,10))

sns.distplot((y_test,pred))
from sklearn.linear_model import Lasso

LassoReg= Lasso()

LassoReg.fit(X_train,y_train)

pred= LassoReg.predict(X_test)

results_df= append_results("Lasso Regression",Lasso(),results_df,y_test,pred)

results_df
figure= plt.figure(figsize=(10,10))

sns.distplot((y_test,pred))
from sklearn.linear_model import ElasticNet

ElasticNetReg= ElasticNet()

ElasticNetReg.fit(X_train,y_train)

pred= ElasticNetReg.predict(X_test)

results_df= append_results("ElasticNet Regression",ElasticNet(),results_df,y_test,pred)

results_df
figure= plt.figure(figsize=(10,10))

sns.distplot((y_test,pred))
results_df.to_csv('resultsEval.csv')