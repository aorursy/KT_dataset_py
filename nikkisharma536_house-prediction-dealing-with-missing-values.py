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
#data = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)
train.plot.scatter(x ='GrLivArea', y= 'SalePrice', ylim = (0,800000)); #, alpha=0.3);
train.plot.scatter(x ='LotFrontage', y= 'SalePrice', ylim = (0,800000)); 
train.plot.scatter(x ='GarageArea', y= 'SalePrice', ylim = (0,800000));
train.columns
train = train.drop(train[(train['OverallQual'] < 5 ) & (train['SalePrice'] > 200000)].index)
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
train = train[train['GarageArea']<1200]
train = train[train['LotFrontage']<200]
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
cate = test.select_dtypes(exclude=[np.number])
cate.describe()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

cols = ('MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
    lbl.fit(list(test[c].values)) 
    test[c] = lbl.transform(list(test[c].values))
train = train.dropna(thresh=0.70*len(train), axis=1)

test = test.dropna(thresh=0.70*len(test), axis=1)
train = train.fillna(train.mean())
test = test.fillna(test.mean())
corr = train.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
y = np.log(train.SalePrice)
X = train.drop(['SalePrice', 'Id'], axis=1)

# Threshold for removing correlated variables
threshold = 0.8

# Absolute value correlation matrix
corr_matrix = X.corr().abs()

# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Remove the columns
X = X.drop(columns = to_drop)
print(X.shape)
print(y.shape)
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def identify_zero_importance_features(X, y, iterations = 2):
    """
    Identify zero importance features in a training dataset based on the 
    feature importances from a gradient boosting model. 
    
    Parameters
    --------
    train : dataframe
        Training features
        
    train_labels : np.array
        Labels for training data
        
    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """
    
    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(X.shape[1])

    # Create the model with several hyperparameters
    model = lgb.LGBMRegressor(objective='regression', boosting_type = 'goss', 
                               n_estimators =550, class_weight = 'balanced')
    
    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):

        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(X, y, 
                                                                            test_size = 0.25, 
                                                                            random_state = i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, 
                  eval_set = [(valid_features, valid_y)])

        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations
    
    feature_importances = pd.DataFrame({'feature': list(X.columns), 
                            'importance': feature_importances}).sort_values('importance', 
                                                                            ascending = False)
    
    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))
    
    return zero_features, feature_importances

zero_features, feature_importances = identify_zero_importance_features(X, y, iterations = 2)
print('zero_features:',zero_features)
print('feature_importances : ', feature_importances)
feature_importances.describe()
#X = X.drop(zero_features, axis = 1)
#test = test.drop(zero_features, axis =1)
pp =np.percentile(feature_importances['importance'], 20) 
print(pp)
to_drop = feature_importances[feature_importances['importance'] <= pp]['feature']
X = X.drop(columns = to_drop)
print(X.shape)
print(y.shape)
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

xg_reg = xgb.XGBRegressor(learning_rate =0.01, n_estimators=5580, 
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
from sklearn.ensemble import GradientBoostingRegressor
gbr_model = GradientBoostingRegressor(n_estimators=6380, learning_rate=0.01,
                                   max_depth=10, max_features='sqrt',
                                   min_samples_leaf=1, min_samples_split=250, 
                                   loss='huber', random_state =6).fit(X_train,y_train)
gbr_preds = gbr_model.predict(X_test)
print('The accuracy of the Gradient boost is',r2_score(y_test,gbr_preds))
print ('RMSE is: \n', mean_squared_error(y_test,gbr_preds))
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
lgbm_model = LGBMRegressor(objective='regression',
                              num_leaves=4,
                              learning_rate=0.05, 
                              n_estimators=1250,
                              max_bin=75, 
                              bagging_fraction=0.8,
                              bagging_freq=9, 
                              feature_fraction=0.45,
                              feature_fraction_seed=9, 
                              bagging_seed=12,
                              min_data_in_leaf=3, 
                              min_sum_hessian_in_leaf=2).fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)
print('The accuracy of the lgbm Regressor is',r2_score(y_test,lgbm_preds))
print ('RMSE is: \n', mean_squared_error(y_test,lgbm_preds))
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
r_alphas =  [0.005, 0.01, 0.05, 1]
ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas = r_alphas, cv =3)).fit(X_train, y_train)

ridge_preds = ridge_model.predict(X_test)
print('The accuracy of the ridge Regressor is',r2_score(y_test,ridge_preds))
print ('RMSE is: \n', mean_squared_error(y_test,ridge_preds))   
    
from sklearn.linear_model import LassoCV

alpha_lasso = np.logspace(-3, -1, 30)


lasso_model = make_pipeline(RobustScaler(),
                             LassoCV(max_iter=1e6,
                                    alphas = alpha_lasso,
                                    random_state = 1)).fit(X_train, y_train)

lasso_preds = lasso_model.predict(X_test)
print('The accuracy of the lasso Regressor is',r2_score(y_test,lasso_preds))
print ('RMSE is: \n', mean_squared_error(y_test,lasso_preds))   
    
from sklearn.linear_model import ElasticNetCV

e_alphas = np.logspace(-3 -2, 30)

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elastic_model= make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e6, alphas=e_alphas, 
                                         l1_ratio=e_l1ratio)).fit(X_train, y_train)

elastic_preds = elastic_model.predict(X_test)
print('The accuracy of the  Elastic Net CV is',r2_score(y_test,elastic_preds))
print ('RMSE is: \n', mean_squared_error(y_test,elastic_preds))   
    
from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#setup models
ridge = make_pipeline(RobustScaler(), 
                      RidgeCV(alphas = r_alphas))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e6,
                                    alphas = alpha_lasso,
                                    random_state = 1)).fit(X_train,y_train)

elasticnet = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e6, alphas=e_alphas, 
                                         l1_ratio=e_l1ratio)).fit(X_train,y_train)

lgbm_model = LGBMRegressor(objective='regression',
                              num_leaves=4,
                              learning_rate=0.05, 
                              n_estimators=1250,
                              max_bin=75, 
                              bagging_fraction=0.8,
                              bagging_freq=9, 
                              feature_fraction=0.45,
                              feature_fraction_seed=9, 
                              bagging_seed=12,
                              min_data_in_leaf=3, 
                              min_sum_hessian_in_leaf=2).fit(X_train,y_train)

gbr_model = GradientBoostingRegressor(n_estimators=6080, learning_rate=0.01,
                                   max_depth=10, max_features='sqrt',
                                   min_samples_leaf=1, min_samples_split=250, 
                                   loss='huber', random_state =6).fit(X_train,y_train)

#stack
stack_gen = StackingCVRegressor(regressors=(ridge,
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
ridge_preds = ridge_model.predict(X_test)
stack_gen_preds = stack_gen_model.predict(X_test)
lgbm_preds = lgbm_model.predict(X_test)
gbr_preds = gbr_model.predict(X_test)
print('The accuracy of the stack Regressor is',r2_score(y_test,stack_gen_preds))
print ('RMSE is: \n', mean_squared_error(y_test,stack_gen_preds))
stack_preds_1 = ((0.1*em_preds) +(0.1*ridge_preds) + (0.1*lasso_preds) + (0.4 * gbr_preds ) 
               + (0.1*lgbm_preds) + (0.2*stack_gen_preds) )
print('The accuracy of the stack Regressor is',r2_score(y_test,stack_preds_1))
print ('RMSE is: \n', mean_squared_error(y_test,stack_preds_1))   
    
#feats = test.select_dtypes(include=[np.number]).interpolate().dropna()
feats = test.drop(['Id'], axis=1)

feats = feats[X_train.columns]
lasso_preds = lasso_model.predict(feats)
em_preds = elastic_model.predict(feats)
#ridge_preds = ridge_model.predict(feats)
stack_gen_preds = stack_gen_model.predict(feats)
lgbm_preds = lgbm_model.predict(feats)
gbr_preds = gbr_model.predict(feats)
stack_preds = ((0.1*em_preds) + (0.1*lasso_preds) + (0.2 * gbr_preds ) 
               + (0.1*lgbm_preds) + (0.5*stack_gen_preds) )
#predictions = model.predict(feats)
final_predictions = np.exp(stack_preds)
print ("Original predictions are: \n", stack_preds[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = final_predictions 
submission.to_csv('submission1.csv', index=False)

