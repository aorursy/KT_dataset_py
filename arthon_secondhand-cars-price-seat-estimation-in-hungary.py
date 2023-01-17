import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

%matplotlib inline
train = pd.read_csv('../input/seat_train.csv')

train.head()
train.info()
train['Year']=train['Year']*12+train['Month']
train.describe().round(decimals=2)
trainShape1=train.shape[0]

train=train.dropna()

trainShape2=train.shape[0]

print('Value of drop NA data in percent: ',(100-(trainShape2/trainShape1)*100))



sns.set(rc={'figure.figsize':(13,7)})

sns.heatmap(train.corr(), annot = True,cmap = "Blues")
correlation = train.corr()

print(correlation['Price'].sort_values(ascending=False).head(5))

print(correlation['Price'].sort_values(ascending=True).head(5))
sns.distplot(train['Price'].dropna(), kde = True, bins = 50)
sns.set_palette("coolwarm")

sns.jointplot(x='Year', y='Price', data=train, kind="hex")
num_columns = train.select_dtypes(exclude='object').columns

corr_to_price = correlation['Price']

n_cols = 4

n_rows = 2

fig, ax_arr = plt.subplots(n_rows, n_cols, figsize=(16,2), sharey=True)

plt.subplots_adjust(bottom=-2.8)

for j in range(n_rows):

    for i in range(n_cols):

        plt.sca(ax_arr[j, i])

        index = i + j*n_cols

        if index < len(num_columns):

            plt.scatter(x=train[num_columns[index]], y=train.Price)

            plt.xlabel(num_columns[index])

            plt.title('Corr to Price = '+ str(np.around(corr_to_price[index])))

plt.show()
from scipy.stats import kurtosis, skew



print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(train.Price)) )

print( 'skewness of normal distribution (should be 0): {}'.format( skew(train.Price) ))
train['Price_log'] = np.log(train['Price'])

sns.distplot(train['Price_log'], kde = True, bins = 50)

print( 'excess kurtosis of normal distribution in Price_log (should be 0): {}'.format( kurtosis(train.Price_log)) )

print( 'skewness of normal distribution in Price_log (should be 0): {}'.format( skew(train.Price_log) ))
train = train.drop(train[(train['ClockStatus'] < 50000)].index)

train = train.drop(train[(train['ClockStatus'] > 500000)].index)



train = train.drop(train[(train['Price'] >4000000)].index)

train = train.drop(train[(train['Price'] <200000)].index)

train.shape
sns.set(rc={'figure.figsize':(13,7)})

sns.set_style("ticks", {"xtick.major.size": 12, "ytick.major.size": 12})

ax = sns.boxplot(x="Type", y="Price", data=train)
xx= pd.get_dummies(train['Type'])

train=train.merge(xx,left_index=True, right_index=True)
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



y = train.Price_log



car_features = ['Year','Engine','Pow1','ClockStatus','ALHAMBRA', 'ALTEA', 'AROSA', 'CORDOBA', 'EXEO', 'IBIZA', 

                'INCA', 'LEON', 'MII','TOLEDO']



X = train[car_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.linear_model import ElasticNet



def inv_y(transformed_y):

    return np.exp(transformed_y)

mae_compare = pd.Series()

mae_compare.index.name = 'Algorithm'



#Random Forest

test_model = RandomForestRegressor(random_state=100, n_estimators=50)

test_model.fit(train_X, train_y)

test_preds = test_model.predict(val_X)

test_mae = mean_absolute_error(inv_y(test_preds), inv_y(val_y))



mae_compare['RandomForest'] = test_mae



#Lasso

lasso_model = Lasso(alpha=0.0005, random_state=5)

lasso_model.fit(train_X, train_y)

lasso_val_predictions = lasso_model.predict(val_X)

lasso_val_mae = mean_absolute_error(inv_y(lasso_val_predictions), inv_y(val_y))



mae_compare['Lasso'] = lasso_val_mae



#Ridge

ridge_model = Ridge(alpha=0.002, random_state=5)

ridge_model.fit(train_X, train_y)

ridge_val_predictions = ridge_model.predict(val_X)

ridge_val_mae = mean_absolute_error(inv_y(ridge_val_predictions), inv_y(val_y))

mae_compare['Ridge'] = ridge_val_mae



#GradientBoosing

gbr_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, 

                                      max_depth=100, random_state=10)

gbr_model.fit(train_X, train_y)

gbr_val_predictions = gbr_model.predict(val_X)

gbr_val_mae = mean_absolute_error(inv_y(gbr_val_predictions), inv_y(val_y))



mae_compare['DradientBoosting'] = gbr_val_mae



#XGBResressor

xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05)

xgb_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X,val_y)], verbose=False)

xgb_val_predictions = xgb_model.predict(val_X)

xgb_val_mae = mean_absolute_error(inv_y(xgb_val_predictions), inv_y(val_y))

mae_compare['XGBRegression'] = xgb_val_mae



#Elasicnet

elastic_net_model = ElasticNet(alpha=0.02, random_state=5, l1_ratio=0.7)

elastic_net_model.fit(train_X, train_y)

elastic_net_val_predictions = elastic_net_model.predict(val_X)

elastic_net_val_mae = mean_absolute_error(inv_y(elastic_net_val_predictions), inv_y(val_y))

mae_compare['ElasticNet'] = elastic_net_val_mae



mae_compare
final_model = Ridge(alpha=0.02, random_state=1010, tol=0.05)

final_model.fit(train_X, train_y)

final_val_predictions = ridge_model.predict(val_X)

final_val_mae = mean_absolute_error(inv_y(ridge_val_predictions), inv_y(val_y))

print('Final prediction MAE with Lasso: ',int(final_val_mae.round(decimals=3)),'HUF')

ridge_model