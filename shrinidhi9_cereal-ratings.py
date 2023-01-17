# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)



dataset = pd.read_csv('../input/cereal.csv')



print(dataset.info())

dataset.drop(['name'],axis=1,inplace=True)

dataset['mfr'] = dataset['mfr'].astype('category')

dataset['type'] = dataset['type'].astype('category')





print(dataset.isnull().sum())

print(dataset.describe(include='all'))



# pip install phik

import phik

from phik import resources, report



corr = dataset.phik_matrix()

plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap



print('Correlation with dependent variable')

corr = dataset.phik_matrix()['rating'].abs()

to_drop_1 = [col for col in corr.index if corr[col]<0.2]

dataset.drop(to_drop_1, axis=1, inplace=True)



corr = dataset.phik_matrix()

plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap



col = corr.index

print('Correlation between independent variables')

for i in range(len(col)):

    for j in range(i+1, len(col)):

        if corr.iloc[i,j] >= 0.8:

            print(f"{col[i]} -{col[j]}")



dataset.drop(['potass','weight'],inplace=True,axis=1)



dataset = pd.get_dummies(dataset)



X = dataset.drop('rating',axis=1).values

y = dataset['rating'].values





from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y,random_state = 7,test_size=0.3)



from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from sklearn.model_selection import cross_val_score



from sklearn.ensemble import RandomForestRegressor

clf_lr = RandomForestRegressor(n_estimators=10)

clf_lr.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,scoring='neg_mean_squared_error')

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





from sklearn.linear_model import LinearRegression

clf_lr = LinearRegression()

clf_lr.fit(train_x , train_y)

accuracies = cross_val_score(estimator = clf_lr, X = train_x, y = train_y, cv = 5,scoring='neg_mean_squared_error')

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




