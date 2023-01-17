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
# import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
% matplotlib inline
# Read and load Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# check index of dataframe
train.columns
train.SalePrice.describe()
#PLot Histogram for 'SalePrice'
sns.distplot(train['SalePrice'])
# Skewness and Kurtosis
print("Skewness : %f" % train['SalePrice'].skew())
print("Kurtosis : %f" % train['SalePrice'].kurt())
target = np.log(train.SalePrice)
print("Skewness : %f" % target.skew())
print("Kurtosis : %f" % target.kurt())
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
corr = numeric_features.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
#'SalePrice' Correlation Matrix
k = 10
cols = corr.nlargest(k , 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale = 1.00)
hm = sns.clustermap(cm , cmap = "Greens",cbar = True,square = True,
                 yticklabels = cols.values, xticklabels = cols.values)
quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
#Analyse SalePrice/GrLiveArea
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)
data.plot.scatter(x ='GrLivArea', y= 'SalePrice', ylim = (0,800000)); #, alpha=0.3);
train = train[train['GarageArea'] < 1200]

# Histogram and normal probability plot
sns.distplot(train['SalePrice'], fit = norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot = plt)
train['Total_Bathrooms'] = (train['FullBath'] + (0.5*train['HalfBath']) +
                            train['BsmtFullBath'] + (0.5*train['BsmtHalfBath']))

test['Total_Bathrooms'] = (test['FullBath'] + (0.5*test['HalfBath']) + 
                           test['BsmtFullBath'] + (0.5*test['BsmtHalfBath']))

train['TotalSF'] = (train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF'])
test['TotalSF'] = (test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF'])



train = train.drop(['FullBath','HalfBath','BsmtFullBath','BsmtHalfBath'], axis=1)
test = test.drop(['FullBath','HalfBath','BsmtFullBath','BsmtHalfBath'], axis=1)

train = train.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis=1)
test = test.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis=1)


# Missing Data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(20)
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()
cate = test.select_dtypes(exclude=[np.number])
cate.describe()
train['Bsmt'] = (train['BsmtQual'] + train['BsmtCond']) 
test['Bsmt'] = (test['BsmtQual'] + test['BsmtCond']) 

train['Garage'] = (train['GarageQual'] + train['GarageCond']) 
test['Garage'] = (test['GarageQual'] + test['GarageCond']) 

train['External'] = (train['ExterQual'] + train['ExterCond']) 
test['External'] = (test['ExterQual'] + test['ExterCond']) 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

cols = ('FireplaceQu', 'Bsmt', 'Garage', 
        'External', 'KitchenQual', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
         'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold','YearRemodAdd')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
    lbl.fit(list(test[c].values)) 
    test[c] = lbl.transform(list(test[c].values))
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
test = test.select_dtypes(include=[np.number]).interpolate().dropna()
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Linear Regression
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
print('The accuracy of the Linear Regression is',r2_score(y_test,y_pred))
print ('RMSE is: \n', mean_squared_error(y_test, y_pred))
import xgboost as xgb

xg_reg = xgb.XGBRegressor(learning_rate =0.01, n_estimators=3460, 
                                     max_depth=3,min_child_weight=0 ,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective= 'reg:linear',nthread=4,
                                     scale_pos_weight=1,seed=27, 
                                     reg_alpha=0.00006)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
print('The accuracy of the xgboost is',r2_score(y_test,preds))
print ('RMSE is: \n', mean_squared_error(y_test,preds))
from sklearn.model_selection import KFold
n_splits_val = 3
kfolds = KFold(n_splits=n_splits_val, shuffle=False)
from sklearn.ensemble import GradientBoostingRegressor
gbr_model = GradientBoostingRegressor(n_estimators=3460, learning_rate=0.01,
                                   max_depth=3, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5).fit(X_train,y_train)
gbr_preds = gbr_model.predict(X_test)
print('The accuracy of the Gradient boost is',r2_score(y_test,gbr_preds))
print ('RMSE is: \n', mean_squared_error(y_test,gbr_preds))
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11).fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)
print('The accuracy of the lgbm Regressor is',r2_score(y_test,lgbm_preds))
print ('RMSE is: \n', mean_squared_error(y_test,lgbm_preds))
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
r_alphas = [.0001, .0003, .0005, .0007, .0009, 
          .01, 0.05, 0.1, 0.3, 1]
ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas = r_alphas,
                                    cv=kfolds)).fit(X_train, y_train)

ridge_preds = ridge_model.predict(X_test)
print('The accuracy of the ridge Regressor is',r2_score(y_test,ridge_preds))
print ('RMSE is: \n', mean_squared_error(y_test,ridge_preds))   
    
from sklearn.linear_model import LassoCV

alphas2 = [1, 0.1, 0.001, 0.0005]
lasso_model = make_pipeline(RobustScaler(),
                             LassoCV(max_iter=1e7,
                                    alphas = alphas2,
                                    random_state = 42, cv = kfolds)).fit(X_train, y_train)

lasso_preds = lasso_model.predict(X_test)
print('The accuracy of the lasso Regressor is',r2_score(y_test,lasso_preds))
print ('RMSE is: \n', mean_squared_error(y_test,lasso_preds))   
    
from sklearn.linear_model import ElasticNetCV

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elastic_model= make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                        cv=kfolds, l1_ratio=e_l1ratio)).fit(X_train, y_train)

elastic_preds = elastic_model.predict(X_test)
print('The accuracy of the  Elastic Net CV is',r2_score(y_test,elastic_preds))
print ('RMSE is: \n', mean_squared_error(y_test,elastic_preds))   
    
from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#setup models
#ridge = make_pipeline(RobustScaler(), 
#                      RidgeCV(alphas = r_alphas))
#
lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas = alphas2,
                              random_state = 42))

elasticnet = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                         l1_ratio=e_l1ratio))

lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, 
                                      min_sum_hessian_in_leaf = 11).fit(X_train,y_train)
gbr_model = GradientBoostingRegressor().fit(X_train,y_train)

#stack
stack_gen = StackingCVRegressor(regressors=(#ridge,
                                            lasso, elasticnet, gbr_model,
                                             lgbm_model), 
                               meta_regressor=gbr_model,
                               use_features_in_secondary=True)

#prepare dataframes
stackX = np.array(X_train)
stacky = np.array(y_train)
stack_gen_model = stack_gen.fit(stackX, stacky)
em_preds = elastic_model.predict(X_test)
lasso_preds = lasso_model.predict(X_test)
#ridge_preds = ridge_model.predict(X_test)
stack_gen_preds = stack_gen_model.predict(X_test)
lgbm_preds = lgbm_model.predict(X_test)
gbr_preds = gbr_model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test,stack_gen_preds))
stack_preds_1 = ((0.1*em_preds) + (0.1*lasso_preds) + (0.4 * gbr_preds ) 
               + (0.2*lgbm_preds) + (0.2*stack_gen_preds))
print('The accuracy of the stack Regressor is',r2_score(y_test,stack_preds_1))
print ('RMSE is: \n', mean_squared_error(y_test,stack_preds_1))   
    
feats = test.select_dtypes(include=[np.number]).interpolate().dropna()
feats = test.drop([ 'Id'], axis=1)
em_preds = elastic_model.predict(feats)
lasso_preds = lasso_model.predict(feats)
#ridge_preds = ridge_model.predict(feats)
stack_gen_preds = stack_gen_model.predict(feats)
lgbm_preds = lgbm_model.predict(feats)
gbr_preds = gbr_model.predict(feats)
stack_preds=((0.1*em_preds) + (0.1*lasso_preds) + (0.4 * gbr_preds ) 
               + (0.2*lgbm_preds) + (0.2*stack_gen_preds))
#predictions = model.predict(feats)
final_predictions = np.exp(stack_preds)
print ("Original predictions are: \n", stack_preds[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = final_predictions 
submission.to_csv('submission1.csv', index=False)

