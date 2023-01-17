# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print('Shape:  {}'.format(train.shape))
print('Columns: {}'.format(train.columns))

sale_price = train['SalePrice']

print('Analyis of Sales Price')
print('Median sale price {}'.format(sale_price.median()))
print('Mean sale price {}'.format(sale_price.mean()))
print('Mode sale price {}'.format(sale_price.mode()))
print('Standard deviation sale price {}'.format(sale_price.std()))
print('Minimum sale price {}'.format(sale_price.min()))
print('Maximum sale price {}'.format(sale_price.max()))
print('Skew for sale price {}'.format(sale_price.skew()))
print('Null records for sale price {}'.format(sale_price.isnull().sum()))

sale_price_log = np.log(sale_price)
print('Skew for log of sale price {}'.format(sale_price_log.skew()))

plt.subplot(1, 2, 1)
plt.title('Sales Price')
plt.hist(sale_price)

plt.subplot(1, 2, 2)
plt.title('Log of Sales Price')
plt.hist(sale_price_log)
plt.show()

plt.subplot(1, 1, 1)
plt.title('Box plot')
plt.boxplot(train['SalePrice'])
plt.show()
print('Analyis of OverallQual')
overall_qual = train['OverallQual']

print('Median overall qual {}'.format(overall_qual.median()))
print('Mean overall qual {}'.format(overall_qual.mean()))
print('Mode overall qual{}'.format(overall_qual.mode()))
print('Standard deviation overall qual {}'.format(overall_qual.std()))
print('Minimum overall qual {}'.format(overall_qual.min()))
print('Maximum overall qual {}'.format(overall_qual.max()))
print('Skew for overall qual {}'.format(overall_qual.skew()))
print('Null records for overall qual {}'.format(overall_qual.isnull().sum()))

plt.subplot(1, 2, 1)
plt.title('Overall Quality')
plt.hist(overall_qual)
print('Analyis of GrLivArea')
grLiv_area = train['GrLivArea']
grLiv_area_log = np.log(grLiv_area)

print('Median GrLivArea {}'.format(grLiv_area.median()))
print('Mean GrLivArea {}'.format(grLiv_area.mean()))
print('Mode GrLivArea {}'.format(grLiv_area.mode()))
print('Standard deviation GrLivArea {}'.format(grLiv_area.std()))
print('Minimum GrLivArea {}'.format(grLiv_area.min()))
print('Maximum GrLivArea {}'.format(grLiv_area.max()))
print('Skew for GrLivArea {}'.format(grLiv_area.skew()))
print('Null records for GrLivArea {}'.format(grLiv_area.isnull().sum()))


plt.subplot(1, 2, 1)
plt.title('GrLivArea')
plt.hist(grLiv_area)

plt.subplot(1, 2, 2)
plt.title('Log of GrLivArea')
plt.hist(grLiv_area_log)
print('Analyis of Garage Cars')
garage_cars = train['GarageCars']

print('Median Garage Cars {}'.format(garage_cars.median()))
print('Mean Garage Cars {}'.format(garage_cars.mean()))
print('Mode Garage Cars {}'.format(garage_cars.mode()))
print('Standard deviation Garage Cars {}'.format(garage_cars.std()))
print('Minimum Garage Cars {}'.format(garage_cars.min()))
print('Maximum Garage Cars {}'.format(garage_cars.max()))
print('Skew for Garage Cars {}'.format(garage_cars.skew()))
print('Null records for Garage Cars {}'.format(garage_cars.isnull().sum()))


plt.subplot(1, 2, 1)
plt.title('Garage Cars')
plt.hist(garage_cars)
plt.figure(figsize = (35,35))
cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)


cor_target = abs(cor["SalePrice"])
relevent_features = cor_target[cor_target>0.5]
relevent_features.sort_values()
X = overall_qual.values.reshape(-1, 1)
y = sale_price_log.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

print('R2 Score for single feature {}'.format(regressor.score(X_test, y_test)))

df = pd.DataFrame({'Actual': np.exp(y_test.flatten()), 'Predicted': np.exp(y_pred.flatten())})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
X = pd.concat([overall_qual, grLiv_area_log], 1)

print('VIF analysis for two features')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif.round(1))



#X = overall_qual.values.reshape(-1, 1)
y = sale_price_log.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

print('R2 Score for two features {}'.format(regressor.score(X_test, y_test)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))

df = pd.DataFrame({'Actual': np.exp(y_test.flatten()), 'Predicted': np.exp(y_pred.flatten())})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



print('Three features')

X = pd.concat([overall_qual, grLiv_area_log, garage_cars], 1)
y = sale_price_log.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

print('R2 Score for two features {}'.format(regressor.score(X_test, y_test)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
ridge_regressor = Ridge() 
ridge_regressor.fit(X_train, y_train)
y_pred = ridge_regressor.predict(X_test)

print('R2 Score for two features {}'.format(ridge_regressor.score(X_test, y_test)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
lasso_regressor = Lasso()
lasso_regressor.fit(X_train, y_train)
y_pred = lasso_regressor.predict(X_test)

print('R2 Score for two features {}'.format(lasso_regressor.score(X_test, y_test)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

print('R2 Score for two features {}'.format(tree_model.score(X_test, y_test)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print('R2 Score for two features {}'.format(rf_model.score(X_test, y_test)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
X = pd.concat([test_data['OverallQual'], np.log(test_data['GrLivArea']), test_data['GarageCars']], 1)
X.fillna


X = pd.concat([test_data['OverallQual'], np.log(test_data['GrLivArea']), test_data['GarageCars']], 1)
X = X.fillna(0)
test_prediction = regressor.predict(X)
pd.Series(test_prediction.flatten())
d= {'Id': test_data['Id'], 'SalePrice': pd.Series(np.exp(test_prediction.flatten()))}
submission_df = pd.DataFrame(d)
submission_df.to_csv('submission.csv' , index=False)
