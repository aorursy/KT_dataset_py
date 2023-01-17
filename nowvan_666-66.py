import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.feature_selection import RFE, f_regression

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)

from sklearn.preprocessing import MinMaxScaler

import scipy as sp  

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



from sklearn.cross_validation import cross_val_score
train = pd.read_csv("../input/kc_house_data.csv")
train.describe(include='all')
# 資料型態

print(train.dtypes)
with sns.plotting_context("notebook",font_scale=2.5):

    g = sns.pairplot(train[['grade','zipcode','price','sqft_living','condition']], 

                 hue='condition', palette='tab20',size=6)

g.set(xticklabels=[]);
str_list = [] # empty list to contain columns with strings (words)

for colname, colvalue in train.iteritems():

    if type(colvalue[1]) == str:

         str_list.append(colname)

# Get to the numeric columns by inversion            

num_list = train.columns.difference(str_list) 

# Create Dataframe containing only numerical features

train_num = train[num_list]

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation of features')

# Draw the heatmap using seaborn

#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)

sns.heatmap(train_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)
from sklearn.model_selection import train_test_split





# 要預測的target是price

target = train["price"]



# 把不需要的feature去除

train = train.drop(['price','date','id'], axis=1)





# 使用dummies將資料拆分

categorial_cols = ['zipcode']



for cc in categorial_cols:

    dummies = pd.get_dummies(train[cc], drop_first=False)

    dummies = dummies.add_prefix("{}#".format(cc))

    train = train.join(dummies)





# 新增新的變數是否有地下室 還有是否有裝修過

train['basement_present'] = train['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) 

train['renovated'] = train['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

train = train.drop(['sqft_basement','yr_renovated'], axis=1)



# 將sqft_living的比重放大

train['sqft_living_squared'] = train['sqft_living'].apply(lambda x: x**2)

train = train.drop(['sqft_living'], axis=1)





#train = train.drop(['sqft_lot','sqft_lot15','yr_built','lat','long'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 0)

train.columns
x_train.to_csv('x_train.csv', index=False)

x_test.to_csv('x_test.csv', index=False)

y_train.to_csv('y_train.csv', index=False)

y_test.to_csv('y_test.csv', index=False)
# Bayesian ridge regression

from sklearn.linear_model import BayesianRidge



brr = BayesianRidge()

brr.fit(x_train, y_train)

cross_rmse_brr = np.mean(np.sqrt(-cross_val_score(brr, x_train, y_train, cv=6, scoring='neg_mean_squared_error')))

cross_r2_brr = np.mean(cross_val_score(brr, x_train, y_train, cv=6, scoring='r2'))

print ('cross_val_score RMSE is : %.0f' %(cross_rmse_brr))

print ('cross_val_score R^2 is : %.6f' %(cross_r2_brr))
# LinearRegression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

cross_rmse_lr = np.mean(np.sqrt(-cross_val_score(lr, x_train, y_train, cv=6, scoring='neg_mean_squared_error')))

cross_r2_lr = np.mean(cross_val_score(lr, x_train, y_train, cv=6, scoring='r2'))

print ('cross_val_score RMSE is : %.0f' %(cross_rmse_lr))

print ('cross_val_score R^2 is : %.6f' %(cross_r2_lr))
# RandomForest

from sklearn.ensemble import RandomForestRegressor



randomforest = RandomForestRegressor(n_estimators=40)

randomforest.fit(x_train, y_train)

cross_rmse_randomforest = np.mean(np.sqrt(-cross_val_score(randomforest, x_train, y_train, cv=6, scoring='neg_mean_squared_error')))

cross_r2_randomforest = np.mean(cross_val_score(randomforest, x_train, y_train, cv=6, scoring='r2'))

print ('cross_val_score RMSE is : %.0f' %(cross_rmse_randomforest))

print ('cross_val_score R^2 is : %.6f' %(cross_r2_randomforest))
# DecisionTree

from sklearn.tree import DecisionTreeRegressor



decisiontree = DecisionTreeRegressor(max_depth=10)

decisiontree.fit(x_train, y_train)

cross_rmse_decisiontree = np.mean(np.sqrt(-cross_val_score(decisiontree, x_train, y_train, cv=6, scoring='neg_mean_squared_error')))

cross_r2_decisiontree = np.mean(cross_val_score(decisiontree, x_train, y_train, cv=6, scoring='r2'))

print ('cross_val_score RMSE is : %.0f' %(cross_rmse_decisiontree))

print ('cross_val_score R^2 is : %.6f' %(cross_r2_decisiontree))
# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsRegressor



knn = KNeighborsRegressor(n_neighbors=10, weights='distance')

knn.fit(x_train, y_train)

cross_rmse_knn = np.mean(np.sqrt(-cross_val_score(knn, x_train, y_train, cv=6, scoring='neg_mean_squared_error')))

cross_r2_knn = np.mean(cross_val_score(knn, x_train, y_train, cv=6, scoring='r2'))

print ('cross_val_score RMSE is : %.0f' %(cross_rmse_knn))

print ('cross_val_score R^2 is : %.6f' %(cross_r2_knn))
# Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor



gbr = GradientBoostingRegressor()

gbr.fit(x_train, y_train)

cross_rmse_gbr = np.mean(np.sqrt(-cross_val_score(gbr, x_train, y_train, cv=6, scoring='neg_mean_squared_error')))

cross_r2_gbr = np.mean(cross_val_score(gbr, x_train, y_train, cv=6, scoring='r2'))

print ('cross_val_score RMSE is : %.0f' %(cross_rmse_gbr))

print ('cross_val_score R^2 is : %.6f' %(cross_r2_gbr))
# Multi-layer Perceptron regressor.

from sklearn.neural_network import MLPRegressor



mlpr = MLPRegressor()

mlpr.fit(x_train, y_train)

cross_rmse_mlpr = np.mean(np.sqrt(-cross_val_score(mlpr, x_train, y_train, cv=6, scoring='neg_mean_squared_error')))

cross_r2_mlpr = np.mean(cross_val_score(mlpr, x_train, y_train, cv=6, scoring='r2'))

print ('cross_val_score RMSE is : %.0f' %(cross_rmse_mlpr))

print ('cross_val_score R^2 is : %.6f' %(cross_r2_mlpr))





y_mlpr = y_test.astype(int).to_frame(name=None).assign(y_pred = mlpr.predict(x_test).astype(int))

# y_pred = mlpr.predict(x_test)

# rmse_mlpr = np.sqrt(mean_squared_error(y_pred,y_test))

# r2_mlpr = r2_score(y_test,y_pred)

# print("R^2 is : %f" %(r2_mlpr))

# print ('RMSE is : %.2f' %(rmse_mlpr))

# y_pred.head(10)

y_mlpr.head(10)
models = pd.DataFrame({

    'AModel': ['Bayesian ridge regression', 'LinearRegression', 'RandomForest', 

              'DecisionTree', 'KNN', 'Gradient Boosting Regressor','MLPRegressor'],

    'Cross Rmse Score': [cross_rmse_brr, cross_rmse_lr, cross_rmse_randomforest, 

              cross_rmse_decisiontree, cross_rmse_knn, cross_rmse_gbr,cross_rmse_mlpr],

    'Cross R2 Score': [cross_r2_brr, cross_r2_lr, cross_r2_randomforest, 

              cross_r2_decisiontree, cross_r2_knn, cross_r2_gbr,cross_r2_mlpr]})

models.sort_values(by='Cross Rmse Score', ascending=True)
y_randomforest = y_test.astype(int).to_frame(name=None).assign(y_pred = randomforest.predict(x_test).astype(int))

y_pred = randomforest.predict(x_test)

rmse_randomforest = np.sqrt(mean_squared_error(y_pred,y_test))

r2_randomforest = r2_score(y_test,y_pred)

print("R^2 is : %f" %(r2_randomforest))

print ('RMSE is : %.2f' %(rmse_randomforest))

y_randomforest.head(10)
plt.scatter(y_randomforest.price, y_randomforest.y_pred)

plt.show()
y_gbr = y_test.astype(int).to_frame(name=None).assign(y_pred = gbr.predict(x_test).astype(int))

y_pred = gbr.predict(x_test)

rmse_gbr = np.sqrt(mean_squared_error(y_pred,y_test))

r2_gbr = r2_score(y_test,y_pred)

print("R^2 is : %f" %(r2_gbr))

print ('RMSE is : %.2f' %(rmse_gbr))

y_gbr.head(10)
plt.scatter(y_gbr.price, y_gbr.y_pred)

plt.show()
y_pred = lr.predict(x_test)

rmse_lr = np.sqrt(mean_squared_error(y_pred,y_test))

r2_lr= r2_score(y_test,y_pred)

print("R^2 is : %f" %(r2_lr))

print ('RMSE is : %.2f' %(rmse_lr))
y_pred = decisiontree.predict(x_test)

rmse_decisiontree = np.sqrt(mean_squared_error(y_pred,y_test))

r2_decisiontree = r2_score(y_test,y_pred)

print("R^2 is : %f" %(r2_decisiontree))

print ('RMSE is : %.2f' %(rmse_decisiontree))
y_pred = brr.predict(x_test)

rmse_decisiontree = np.sqrt(mean_squared_error(y_pred,y_test))

r2_decisiontree = r2_score(y_test,y_pred)

print("R^2 is : %f" %(r2_decisiontree))

print ('RMSE is : %.2f' %(rmse_decisiontree))
