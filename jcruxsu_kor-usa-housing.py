import numpy as np

import os

import pandas as pd
DATA_PATH='/kaggle/input/usa-housing/USA_Housing.csv'
import missingno as msno

import seaborn as sns

import matplotlib.pyplot as plt
housing = pd.read_csv(DATA_PATH)
housing.head()
housing.describe()
housing.isnull().sum()
sns.pairplot(housing)
housing.hist(bins=50,figsize=(20,15))
import scipy as sp



cor_abs = abs(housing.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=6, columns='Price').index 

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(housing[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
housing["Avg. Area Income"].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(housing["Avg. Area Income"],ax=ax)

plt.show()

housing.plot(kind="scatter",x="Avg. Area Income",y="Price",alpha=0.2,cmap= plt.get_cmap("jet"),colorbar=True)
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(housing["Avg. Area House Age"],ax=ax)

plt.show()



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 18))

ax = fig.add_subplot(111, projection='3d')



xs = housing['Avg. Area Income']

ys = housing['Avg. Area House Age']

zs = housing['Price']

ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')



ax.set_xlabel('Avg. Area Income')

ax.set_ylabel('Avg. Area House Age')

ax.set_zlabel('Price')

ax.scatter(xs, ys, zs, c = zs, s= 50, alpha=0.5, cmap=plt.cm.Greens)

#ax.view_init(60, 35)

plt.show()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(housing["Avg. Area Number of Rooms"],ax=ax)

plt.show()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(housing["Avg. Area Number of Bedrooms"],ax=ax)

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 18))

ax = fig.add_subplot(111, projection='3d')



xs = housing['Avg. Area Number of Bedrooms']

ys = housing['Avg. Area Number of Rooms']

zs = housing['Price']

ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')



ax.set_xlabel('Number of Bedrooms')

ax.set_ylabel('Number of Rooms')

ax.set_zlabel('Price')

ax.scatter(xs, ys, zs, c = zs, s= 50, alpha=0.5, cmap=plt.cm.Greens)

#ax.view_init(60, 35)

plt.show()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(housing["Area Population"],ax=ax)

plt.show()
housing['Address'].head()
new_features = ['apt']
aaa =housing['Address'].values
#housing['Address'].loc[housing['Address'].contains(pat='Apt')]

housing_series = pd.Series(housing['Address'])

Apt_index = housing_series[housing_series.str.contains('Apt')].index

print(Apt_index[:10])
housing['IsApt']=0

housing['IsApt'].iloc[Apt_index] = 1
ST_INitial = ['AK','AL','AR','AZ','CA','CO','CT','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']





for st in ST_INitial:

    housing[st]=0

    tmp_st_index = housing_series[housing_series.str.contains(st)].index

    housing[st].iloc[tmp_st_index] =1

    
housing = housing.drop(['Address'],axis=1)
y_label = housing['Price']

housing = housing.drop(['Price'],axis=1)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
rmse_res_list = []

mse_res_list = []

mae_res_list = []

model_list = ['lin_reg','rf_reg']
lin_reg = LinearRegression()

lin_reg.fit(housing,y_label)

lin_predictions = lin_reg.predict(housing)

lin_mse = mean_squared_error(y_label,lin_predictions)

lin_mae = mean_absolute_error(y_label,lin_predictions)

lin_rmse = np.sqrt(lin_mse)
print("root mean_squared_error :",lin_rmse)



print("mean_squared_error : ",lin_mse)



print("mean_absolute_error :",lin_mae)

rmse_res_list.append(lin_rmse)

mse_res_list.append(lin_mse)

mae_res_list.append(lin_mae)
param_grid = [

    {'n_estimators': [50,100],'max_features' : [8,20]},

    {'bootstrap':[False],'max_depth':[5,10]},

]

forest_reg = RandomForestRegressor()

rf_grid_search = GridSearchCV(forest_reg,param_grid,cv=5,

                          scoring='neg_mean_squared_error',return_train_score=True)



rf_grid_search.fit(housing,y_label)
rf_cvres = rf_grid_search.cv_results_

for mean_score, params in zip(rf_cvres["mean_test_score"],rf_cvres["params"]):

    print(np.sqrt(-mean_score),params)
rf_final_res = rf_grid_search.best_estimator_.predict(housing)

print("root mean_squared_error :",np.sqrt(mean_squared_error(rf_final_res,y_label)))

print("mean_squared_error : ", mean_squared_error(rf_final_res,y_label))

print("mean_absolute_error :",mean_absolute_error(rf_final_res,y_label))

rmse_res_list.append(np.sqrt(mean_squared_error(rf_final_res,y_label)))

mse_res_list.append(mean_squared_error(rf_final_res,y_label))

mae_res_list.append(mean_absolute_error(rf_final_res,y_label))
from pandas import Series

fig,ax = plt.subplots(2,2,figsize=(9,5))

plt.figure(figsize=(8, 8))

sns.barplot(model_list,rmse_res_list,ax=ax[0][0])

ax[0][0].set_title("rmse")

sns.barplot(model_list,mse_res_list,ax=ax[0][1])

ax[0][1].set_title("mse")

sns.barplot(model_list,mae_res_list,ax=ax[1][0])

ax[1][0].set_title("mae")

plt.show()
mae_res_list
fig,ax = plt.subplots(1,2,figsize=(10,5))

ax[0].scatter(y_label,rf_final_res);

ax[0].set_title("random forest")

ax[1].scatter(y_label,lin_predictions);

ax[1].set_title("linear regression")
feature_importance = rf_grid_search.best_estimator_.feature_importances_

Series_feat_imp = Series(feature_importance, index=housing.columns)
print(Series_feat_imp)