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
# Missing Data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(20)
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()
print ("Original: \n") 
print (train.Street.value_counts(), "\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
condition_pivot = train.pivot_table(index='SaleCondition',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)
condition_pivot = train.pivot_table(index='KitchenQual',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Kitchen Qual')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
def encode(x): return 1 if x == 'Ex' else 0
train['enc_kitchen'] = train.SaleCondition.apply(encode)
test['enc_kitchen'] = test.SaleCondition.apply(encode)
condition_pivot = train.pivot_table(index='Utilities',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Utilities')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
train['enc_utilities'] = pd.get_dummies(train.Utilities, drop_first=True)
test['enc_utilities'] = pd.get_dummies(train.Utilities, drop_first=True)
condition_pivot = train.pivot_table(index='SaleType',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Type')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
def encode(x): return 1 if x == 'Con' else 0
train['enc_type'] = train.SaleType.apply(encode)
test['enc_type'] = test.SaleType.apply(encode)
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
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

xg_reg = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
print('The accuracy of the xgboost is',r2_score(y_test,preds))
print ('RMSE is: \n', mean_squared_error(y_test,preds))
from sklearn.ensemble import GradientBoostingRegressor
gbr_model = GradientBoostingRegressor().fit(X_train,y_train)
gbr_preds = gbr_model.predict(X_test)
print('The accuracy of the Gradient boost is',r2_score(y_test,gbr_preds))
print ('RMSE is: \n', mean_squared_error(y_test,gbr_preds))
#create dummies
X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)
from sklearn.ensemble import BaggingRegressor
from sklearn import tree
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(X_train, y_train)
bag_preds = model.predict(X_test)
print('The accuracy of the Bagging Regressor is',r2_score(y_test,bag_preds))
print ('RMSE is: \n', mean_squared_error(y_test,bag_preds))
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, 
                                      min_sum_hessian_in_leaf = 11).fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)
print('The accuracy of the Bagging Regressor is',r2_score(y_test,lgbm_preds))
print ('RMSE is: \n', mean_squared_error(y_test,lgbm_preds))
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
r_alphas = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas = r_alphas)).fit(X_train, y_train)

ridge_preds = ridge_model.predict(X_test)
print('The accuracy of the ridge Regressor is',r2_score(y_test,ridge_preds))
print ('RMSE is: \n', mean_squared_error(y_test,ridge_preds))   
    
from sklearn.linear_model import LassoCV

alphas= [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
           0.0006, 0.0007, 0.0008]


lasso_model = make_pipeline(RobustScaler(),
                             LassoCV(max_iter=1e7,
                                    alphas = alphas,
                                    random_state = 42)).fit(X_train, y_train)

lasso_preds = lasso_model.predict(X_test)
print('The accuracy of the lasso Regressor is',r2_score(y_test,lasso_preds))
print ('RMSE is: \n', mean_squared_error(y_test,lasso_preds))   
    
from sklearn.linear_model import ElasticNetCV

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elastic_cv = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                         l1_ratio=e_l1ratio))

elastic_model = elastic_cv.fit(X_train, y_train)
elastic_preds = elastic_model.predict(X_test)
print('The accuracy of the lasso Regressor is',r2_score(y_test,elastic_preds))
print ('RMSE is: \n', mean_squared_error(y_test,elastic_preds))   
    
from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#setup models
ridge = make_pipeline(RobustScaler(), 
                      RidgeCV(alphas = r_alphas))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas = alphas,
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
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr_model,
                                             lgbm_model), 
                               meta_regressor=gbr_model,
                               use_features_in_secondary=True)

#prepare dataframes
stackX = np.array(X_train)
stacky = np.array(y_train)
stack_gen_model = stack_gen.fit(stackX, stacky)
em_preds = elastic_model.predict(X_test)
lasso_preds = lasso_model.predict(X_test)
ridge_preds = ridge_model.predict(X_test)
stack_gen_preds = stack_gen_model.predict(X_test)
lgbm_preds = lgbm_model.predict(X_test)
gbr_preds = gbr_model.predict(X_test)
stack_preds = ((0.1*em_preds) + (0.2*lasso_preds) + (0.1*ridge_preds) + (0.2 * gbr_preds ) 
               + (0.1*lgbm_preds) + (0.3*stack_gen_preds))
print('The accuracy of the stack Regressor is',r2_score(y_test,stack_preds))
print ('RMSE is: \n', mean_squared_error(y_test,stack_preds))   
    
feats = test.select_dtypes(include=[np.number]).interpolate().dropna()
feats = feats.drop([ 'Id'], axis=1)
em_preds = elastic_model.predict(feats)
lasso_preds = lasso_model.predict(feats)
ridge_preds = ridge_model.predict(feats)
stack_gen_preds = stack_gen_model.predict(feats)
lgbm_preds = lgbm_model.predict(feats)
gbr_preds = gbr_model.predict(feats)
stack_preds = ((0.1*em_preds) + (0.2*lasso_preds) + (0.1*ridge_preds) + (0.2 * gbr_preds ) 
               + (0.1*lgbm_preds) + (0.3*stack_gen_preds))
#predictions = model.predict(feats)
final_predictions = np.exp(stack_preds)
print ("Original predictions are: \n", stack_preds[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = final_predictions 
submission.to_csv('submission1.csv', index=False)
