import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)



dataset = pd.read_csv('../input/diamonds.csv', usecols= range(1,11))

print(dataset.info())

# import os

# print(os.listdir("../input"))

print(dataset.isnull().sum())

print(dataset.shape)



dataset = pd.get_dummies(dataset)
dataset['vol'] =dataset['x'] * dataset['y']*dataset['z']



x = dataset.drop(["price",'x','y','z'],axis=1).values

y = dataset['price'].values

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 2,test_size=0.3)





from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from sklearn.model_selection import cross_val_score



# 

# from sklearn.model_selection import GridSearchCV

# params = [{'n_estimators':[15,20,25,30,35,40], 'max_depth':[4,5,6,7,8,9,10]}]

# vc = GridSearchCV(estimator=RandomForestRegressor(),param_grid=params,

#                   verbose=1,cv=10,n_jobs=-1)

# res = vc.fit(train_x,train_y)

# print(res.best_score_)

# print(res.best_params_)



from sklearn.linear_model import LinearRegression

clf_lr = LinearRegression(0)

clf_lr.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)

y_pred = clf_lr.predict(test_x)

print('')

print('####### LinearRegression #######')

print('Score : %.4f' % clf_lr.score(test_x, test_y))

print(accuracies)



mse = mean_squared_error(test_y, y_pred)

mae = mean_absolute_error(test_y, y_pred)

rmse = mean_squared_error(test_y, y_pred)**0.5

r2 = r2_score(test_y, y_pred)



print('')

print('MSE    : %0.2f ' % mse)

print('MAE    : %0.2f ' % mae)

print('RMSE   : %0.2f ' % rmse)

print('R2     : %0.2f ' % r2)

clf_lr = RandomForestRegressor(n_estimators=10)

clf_lr.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,verbose = 1)

y_pred = clf_lr.predict(test_x)

print('')

print('####### RandomForestRegressor #######')

print('Score : %.4f' % clf_lr.score(test_x, test_y))

print(accuracies)



mse = mean_squared_error(test_y, y_pred)

mae = mean_absolute_error(test_y, y_pred)

rmse = mean_squared_error(test_y, y_pred)**0.5

r2 = r2_score(test_y, y_pred)



print('')

print('MSE    : %0.2f ' % mse)

print('MAE    : %0.2f ' % mae)

print('RMSE   : %0.2f ' % rmse)

print('R2     : %0.2f ' % r2)
