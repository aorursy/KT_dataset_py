import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Read csv file

df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.info()
# Correlation

Cor_heat = df.corr()
plt.figure(figsize=(16,16))
sns.heatmap(Cor_heat, cmap = "RdBu_r", vmax=0.9, square=True)
## Lets see what most important features we have

IF = Cor_heat['price'].sort_values(ascending=False).head(10).to_frame()
IF.head(5)
# Split the Data

Feature_data = df.drop(['price','date', 'id'], axis=1)
Target_data = df['price']
# Check Target Data Skewness

print('Skew Value : ' + str(Target_data.skew()))
sns.distplot(Target_data)
# transform target

from scipy.special import inv_boxcox
from scipy.stats import boxcox
f = plt.figure(figsize=(16,16))

# log 1 Transform
ax = f.add_subplot(221)
L1p = np.log1p(Target_data)
sns.distplot(L1p,color='b',ax=ax)
ax.set_title('skew value Log 1 transform: ' + str(np.log1p(Target_data).skew()))

# Square Log Transform
ax = f.add_subplot(222)
SRT = np.sqrt(Target_data)
sns.distplot(SRT,color='c',ax=ax)
ax.set_title('Skew Value Square Transform: ' + str(np.sqrt(Target_data).skew()))

# Log Transform
ax = f.add_subplot(223)
LT = np.log(Target_data)
sns.distplot(LT, color='r',ax=ax)
ax.set_title('Skew value Log Transform: ' + str(np.log(Target_data).skew()))

# Box Cox Transform
ax = f.add_subplot(224)
BCT,fitted_lambda = boxcox(Target_data,lmbda=None)
sns.distplot(BCT,color='g',ax=ax)
ax.set_title('Skew Value Box Cox Transform: ' + str(pd.Series(BCT).skew()))
Target_data = BCT
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
# I'm using 5 fold in this cross val score

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, Feature_data, Target_data, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
# Initiate Models
## Gradient Boosting

GB = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
## XGBoost

XGB = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=220,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
## LightGBM

LGB = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=320,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model = [GB, XGB, LGB]

for x in model:    
    score = rmse_cv(x).mean()
    print('RMSE Score with ' + str(x.__class__.__name__) + ' : ' + str(score))
Predicted_data = []

for x in model:
    result = cross_val_predict(x, Feature_data, Target_data, cv=5)
    Predicted_data.append(result)
f = plt.figure(figsize=(15,5))

# Gradient Boosting
ax = f.add_subplot(131)
sns.distplot(Target_data, hist=False, label="Actual Values")
sns.distplot(Predicted_data[0], hist=False, label="Predicted Values")
ax.set_title('Distribution Comaprison with Gradient Boosting')

# XGBoost
ax = f.add_subplot(132)
sns.distplot(Target_data, hist=False, label="Actual Values")
sns.distplot(Predicted_data[1], hist=False, label="Predicted Values")
ax.set_title('Distribution Comaprison with XGBoost')

# LightGBM
ax = f.add_subplot(133)
sns.distplot(Target_data, hist=False, label="Actual Values")
sns.distplot(Predicted_data[2], hist=False, label="Predicted Values")
ax.set_title('Distribution Comaprison with LightGBM')

plt.show()