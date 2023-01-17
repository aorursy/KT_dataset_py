# Import Libraries 

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os,time,sys
import lightgbm as lgb
os.chdir("../input")
Tr = pd.read_csv("train.csv")
Tr
Tr.info()
test= pd.read_csv("test.csv")
test
Tr.SalePrice.describe()
print ("Skew is:", Tr.SalePrice.skew())
plt.hist(Tr.SalePrice, color='blue')
plt.show()
target = np.log(Tr.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
numeric_features = Tr.select_dtypes(include=[np.number])
numeric_features.dtypes
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
Tr.OverallQual.unique()
quality_pivot = Tr.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)

quality_pivot
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
nulls = pd.DataFrame(Tr.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
categoricals = Tr.select_dtypes(exclude=[np.number])
categoricals.describe()
data = Tr.select_dtypes(include=[np.number]).interpolate().dropna()
y = np.log(Tr.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()
MAXDEPTH = 60
regr = RandomForestRegressor(n_estimators=1200,   # No of trees in forest
                             criterion = "mse",       
                             max_features = "sqrt",   # no of features to consider for the best split
                             max_depth= MAXDEPTH,     #  maximum depth of the tree
                             min_samples_split= 2,    # minimum number of samples required to split an internal node
                             min_impurity_decrease=0, # Split node if impurity decreases greater than this value.
                             oob_score = True,        # whether to use out-of-bag samples to estimate error on unseen data.
                             n_jobs = -1,             #  No of jobs to run in parallel
                             random_state=0,
                             verbose = 10             # Controls verbosity of process
                             )

regr.fit(X_train, y_train)
predictions = regr.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
print ("R^2 is: \n", model.score(X_test, y_test))
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Random Forest Model')
plt.show()
Output = pd.DataFrame(data=predictions,columns = ['SalePrice'])
Output['Id'] = test.Id
Output = Output[['Id','SalePrice']]
Output
submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)
print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions
submission