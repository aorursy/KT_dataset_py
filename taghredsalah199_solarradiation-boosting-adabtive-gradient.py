import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_solar= pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')

df_solar
sns.distplot(df_solar['UNIXTime'])
df_solar=df_solar.drop(['Data','Time','TimeSunRise','TimeSunSet'],axis=1)

df_solar
figure= plt.figure(figsize=(10,10))

sns.heatmap(df_solar.corr(),annot=True)
figure= plt.figure(figsize=(20,10))

sns.pairplot(df_solar)
X= df_solar.drop('Radiation',axis=1)

y=df_solar['Radiation']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
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
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor

ada_reg= AdaBoostRegressor(RandomForestRegressor())

ada_reg.fit(X_train,y_train)
pred = ada_reg.predict(X_test)
results_df= append_results("AdaBoost Regression",AdaBoostRegressor(RandomForestRegressor()),results_df,y_test,pred)

results_df
sns.distplot((y_test,pred))
from sklearn.ensemble import GradientBoostingRegressor

gr_reg= GradientBoostingRegressor()

gr_reg.fit(X_train,y_train)

pred = gr_reg.predict(X_test)
sns.distplot((y_test,pred))
results_df= append_results("Gradient Regression",GradientBoostingRegressor(),results_df,y_test,pred)

results_df