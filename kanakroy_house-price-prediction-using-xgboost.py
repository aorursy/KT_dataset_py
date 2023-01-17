# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics
df_train=pd.read_csv('../input/train.csv')
df_train.info()
df_train.describe()
df_test=pd.read_csv('../input/test.csv')
df_test.info()
# Saving the Id columns
train_id=df_train['Id']
test_id=df_test['Id']

# Dropping Id columns from both train and test as these are not needed for prediction
df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)
# Exploring outliers

fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# Removing outliers

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000)].index)

#Check the scatter plot again
fig, ax = plt.subplots()
ax.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# Histogram plot of SalePrice
sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
# Probability plot or QQ plot to see the linear fit of the SalePrice

fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
# Log Transformation of the Target Variable
df_train["SalePrice"]=np.log1p(df_train["SalePrice"]) # log(1+x)
# Plots after transformation

# Histogram plot of SalePrice
sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# Probability plot or QQ plot to see the linear fit of the SalePrice
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
# Combining test data and train data
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data = pd.get_dummies(all_data)
all_data.shape
# Imputing missing data by the mean of each column.
all_data = all_data.fillna(all_data.mean())
# Generating train and test sets

X_train=all_data[:df_train.shape[0]]
X_test=all_data[:df_test.shape[0]]

y_train=df_train.SalePrice
# Importing libraries for modelling

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# Cross Validation Strategy to pick the best model

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
# Linear regression
model_LinearReg = LinearRegression()
model_LinearReg.fit(X_train, y_train)
rmse_LinearReg = rmse_cv(model_LinearReg).mean()
rmse_LinearReg
# RidgeCV
model_RidgeCV = RidgeCV()
model_RidgeCV.fit(X_train, y_train)
rmse_RidgeCV = rmse_cv(model_RidgeCV).mean()
rmse_RidgeCV
# ElasticNet
model_ElasticNet=ElasticNet()
model_ElasticNet.fit(X_train,y_train)
rmse_ElasticNet=rmse_cv(model_ElasticNet).mean()
rmse_ElasticNet
# ElasticNetCV
model_ElasticNetCV=ElasticNetCV()
model_ElasticNetCV.fit(X_train,y_train)
rmse_ElasticNetCV=rmse_cv(model_ElasticNetCV).mean()
rmse_ElasticNetCV
# lassoCV
model_lassoCV=LassoCV()
model_lassoCV.fit(X_train,y_train)
rmse_lassoCV=rmse_cv(model_lassoCV).mean()
rmse_lassoCV
# LassoLarsCV
model_LassoLarsCV=LassoLarsCV()
model_LassoLarsCV.fit(X_train,y_train)
rmse_LassoLarsCV=rmse_cv(model_LassoLarsCV).mean()
rmse_LassoLarsCV
# KernelRidge
model_KernelRidge=KernelRidge()
model_KernelRidge.fit(X_train,y_train)
rmse_KernelRidge=rmse_cv(model_KernelRidge).mean()
rmse_KernelRidge
# RandomForestRefressor

model_RandomForest=RandomForestRegressor()
model_RandomForest.fit(X_train,y_train)
rmse_RandomForest=rmse_cv(model_RandomForest).mean()
rmse_RandomForest
# XGBoost
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y_train)
rmse_xgb = model.values[-1,0]
rmse_xgb
# Creating RMSE Dictionary
rmse_dict = {'data':[rmse_RidgeCV, rmse_ElasticNet, rmse_ElasticNetCV, rmse_lassoCV,rmse_LassoLarsCV,rmse_KernelRidge,rmse_RandomForest,rmse_xgb]}

# Creating RMSE DataFrame 
rmse_df = pd.DataFrame(data = rmse_dict, index = ['RidgeCV','ElasticNet','ElasticNetCV','LassoCV','LassoLarsCV','KernelRidge','RandomForest','XGBoost'])

# Plotting RMSE 
rmse_df.plot.bar(legend = False, title = 'Root Mean Square Error')
y_test = model_xgb.predict(X_test)
hprice = pd.DataFrame({"id":test_id})
hprice = hprice.assign(SalePrice = y_test)
hprice.head()
# Feeding Id and SalePrice into Test data
df_test['SalePrice']=y_test
df_test['Id']=test_id
# Visualizing SalePrice with respect to GrLivArea

fig, ax = plt.subplots()
ax.scatter(x = df_test['GrLivArea'], y = df_test['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# Histogram plot of SalePrice
sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
# Histogram plot of SalePrice
sns.distplot(df_test['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_test['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
df_test[['Id', 'SalePrice']].to_csv('Predicted_House_Price.csv', index=False)