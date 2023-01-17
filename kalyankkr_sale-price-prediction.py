# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pds.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from statistics import mode
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.linear_model import Ridge, Lasso,ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from catboost import Pool, CatBoostRegressor, cv
import xgboost as xgb
import lightgbm as lgb
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.shape,test.shape
train.describe()
train_ID = train['Id']
test_ID = test['Id']

# Now drop the 'Id' colum since it's unnecessary for the prediction process
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
sns.regplot(train["GrLivArea"],y=train["SalePrice"],fit_reg=True)
plt.show()

# Removing two very extreme outliers in the bottom right hand corner
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# Re-check graph
sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=True)
plt.show()

(mu,sigma)=norm.fit(train.SalePrice)
sns.distplot(train.SalePrice,fit=norm)
plt.legend(["$\mu=$ {:.2f} and $\sigma=$ {:.2f}".format(mu,sigma)],loc="best")
train.SalePrice = np.log1p(train.SalePrice)
(mu,sigma)=norm.fit(train.SalePrice)
sns.distplot(train.SalePrice,fit=norm)
plt.legend(["$\mu=$ {:.2f} and $\sigma=$ {:.2f}".format(mu,sigma)],loc="best")
train_nS=train.shape[0]
test_nS=test.shape[0] # shpaes of train and tests for sperating them back

train_y=train.SalePrice.values
full_data=pd.concat((train,test)).reset_index(drop=True) #concating the train and test sets

full_data.drop(["SalePrice"],axis=1,inplace=True) #dropping the target values

full_data.shape
missing_data_rank=(full_data.isnull().sum()/len(full_data))*100
print("total number of columns with values misiing : {}".format(missing_data_rank[missing_data_rank>0].count()))
missed =pd.DataFrame({"Missing Percentage": missing_data_rank[missing_data_rank>0].sort_values(ascending =False)})


missed_features=list(missed.index)
full_data.head(30)
full_data.GarageQual.unique()
# All columns where missing values can be replaced with 'None'
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    full_data[col] = full_data[col].fillna('None')

# All columns where missing values can be replaced with 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    full_data[col] = full_data[col].fillna(0)


# All columns where missing values can be replaced with the mode (most frequently occurring value)
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):
    full_data[col] = full_data[col].fillna(full_data[col].mode()[0])

# Imputing LotFrontage with the median (middle) value
full_data['LotFrontage'] = full_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']

missing_data=(full_data.isnull().sum()/len(full_data))*100
print("total number of columns with values misiing : {}".format(missing_data[missing_data>0].count()))
missed =pd.DataFrame({"Missing Percentage": missing_data[missing_data>0].sort_values(ascending =False)})

full_data.info()
# Converting those variables which should be categorical, rather than numeric
for col in ('MSSubClass', 'OverallCond', 'YrSold', 'MoSold'):
    full_data[col] = full_data[col].astype(str)
    
full_data.info()
# Applying a log(1+x) transformation to all skewed numeric features
numeric_feats = full_data.dtypes[full_data.dtypes != "object"].index

# Compute skewness
skewed_feats = full_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)
# Check on number of skewed features above 75% threshold
skewness = skewness[abs(skewness) > 0.75]
print("Total number of features requiring a fix for skewness is: {}".format(skewness.shape[0]))
# Now let's apply the box-cox transformation to correct for skewness
skewed_features = skewness.index
lam = 0.15
for feature in skewed_features:
    full_data[feature] = boxcox1p(full_data[feature], lam)
full_data = full_data.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)
#highlyrepeated_values= [col for col in full_data.select_dtypes(exclude=['number']) if 1 - sum(full_data[col] == mode(full_data[col]))/len(full_data) < 0.03]
# Dropping these columns from both datasets
#full_data = full_data.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)
full_data.info()
obj_features=list(full_data.select_dtypes(include="object").columns)
len(obj_features)
# List of columns to Label Encode
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# Process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(full_data[c].values)) 
    full_data[c] = lbl.transform(list(full_data[c].values))

# Check on data shape        
print('Shape all_data: {}'.format(full_data.shape))
full_data.info()
full_data=pd.get_dummies(full_data)
full_data.shape
full_data.i

# Now to return to separate train/test sets for Machine Learning
train_x = full_data[:train_nS]
test_x= full_data[train_nS:]
# Defining two rmse_cv functions

def rmse_cv(model):
    
    rmse=np.sqrt(-cross_val_score(model, train_x,train_y,scoring="neg_mean_squared_error",cv=10))
    return (rmse)

alphas = [0.05, 0.1, 0.3, 1, 3, 5,7, 10, 15, 30]
#alphas=np.arange(0.05,30,0.05)
# Iterate over alpha's
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
print(cv_ridge)
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")

# 5 looks like the optimal alpha level, so let's fit the Ridge model with this value
#model_ridge = Ridge(alpha = 10)
model_ridge = Ridge(alpha = 7)
alphas = [0.01, 0.005, 0.001, 0.0002,0.0003,0.0004,0.0005,0.0001]
#alphas=np.arange(0.0001,0.01,0.0005)
# Iterate over alpha's
cv_lasso = [rmse_cv(Lasso(alpha = alpha,random_state=1)).mean() for alpha in alphas]

# Plot findings
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")
print(cv_lasso)
# Initiating Lasso model
model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0004))
# Setting up list of alpha's
alphas = [0.01, 0.005, 0.001, 0.00055,0.0006, 0.0001]
#alphas=np.arange(0.0001,1,0.0004)
# Iterate over alpha's
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_elastic = pd.Series(cv_elastic, index = alphas)
cv_elastic.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")
print(cv_elastic)
# Initiating ElasticNet model
model_elasticnet = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0006))
# Setting up list of alpha's
alphas = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# Iterate over alpha's
cv_krr = [rmse_cv(KernelRidge(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_krr = pd.Series(cv_krr, index = alphas)
cv_krr.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")
print(cv_krr)
# Initiatiing KernelRidge model
model_krr = make_pipeline(RobustScaler(), KernelRidge(alpha=7, kernel='polynomial', degree=2.65, coef0=6.9))
# Initiating Gradient Boosting Regressor
model_gbr = GradientBoostingRegressor(n_estimators=1200, 
                                      learning_rate=0.05,
                                      max_depth=4, 
                                      max_features='sqrt',
                                      min_samples_leaf=15, 
                                      min_samples_split=10, 
                                      loss='huber',
                                      random_state=5)
cv_gbr=rmse_cv(model_gbr).mean()
cv_gbr


# Initiating XGBRegressor
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2,
                             learning_rate=0.025,
                             max_depth=3,
                             n_estimators=1550)
cv_xgb = rmse_cv(model_xgb).mean()
cv_xgb
# Initiating LGBMRegressor model
model_lgb = lgb.LGBMRegressor(objective='regression',
                              num_leaves=4,
                              learning_rate=0.05, 
                              n_estimators=1080,
                              max_bin=75, 
                              bagging_fraction=0.80,
                              bagging_freq=5, 
                              feature_fraction=0.232,
                              feature_fraction_seed=9, 
                              bagging_seed=9,
                              min_data_in_leaf=6, 
                              min_sum_hessian_in_leaf=11)
cv_lgb = rmse_cv(model_lgb).mean()
cv_lgb
# Fitting all models with rmse_cv function, apart from CatBoost
cv_ridge = rmse_cv(model_ridge).mean()
cv_lasso = rmse_cv(model_lasso).mean()
cv_elastic = rmse_cv(model_elasticnet).mean()
cv_krr = rmse_cv(model_krr).mean()
cv_gbr = rmse_cv(model_gbr).mean()
cv_xgb = rmse_cv(model_xgb).mean()
cv_lgb = rmse_cv(model_lgb).mean()


# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['Ridge',
              'Lasso',
              'ElasticNet',
              'Kernel Ridge',
              'Gradient Boosting Regressor',
              'XGBoost Regressor',
              'Light Gradient Boosting Regressor',
              ],
    'Score': [cv_ridge,
              cv_lasso,
              cv_elastic,
              cv_krr,
              cv_gbr,
              cv_xgb,
              cv_lgb]})

# Build dataframe of values
result_df = results.sort_values(by='Score', ascending=True).reset_index(drop=True)
result_df.head(8)


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


#Averaged base models score
averaged_models = AveragingModels(models = (model_elasticnet, model_gbr, model_krr, model_lasso))
score = rmse_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


n_folds = 5
def rmsle_cv(model):
    #kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train_x.values, train_y, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)
#Stacking averaged Models Class
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)




stacked_averaged_models = StackingAveragedModels(base_models = (model_elasticnet, model_gbr, model_krr),meta_model = model_lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

#define a rmsle evaluation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
#Final Training and Prediction
stacked_averaged_models.fit(train_x.values, train_y)
stacked_train_pred = stacked_averaged_models.predict(train_x.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test_x.values))
print(rmsle(train_y, stacked_train_pred))
model_xgb.fit(train_x, train_y)
xgb_train_pred = model_xgb.predict(train_x)
xgb_pred = np.expm1(model_xgb.predict(test_x))
print(rmsle(train_y, xgb_train_pred))


model_lgb.fit(train_x, train_y)
lgb_train_pred = model_lgb.predict(train_x)
lgb_pred = np.expm1(model_lgb.predict(test_x.values))
print(rmsle(train_y, lgb_train_pred))




print('RMSLE score on train data:')
print(rmsle(train_y,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
#ensemble=xgb_pred
# Create stacked model
#stacked = (lasso_pred + elastic_pred + ridge_pred + xgb_pred + lgb_pred + krr_pred + gbr_pred) / 7
# Setting up competition submission
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble #stacked
sub.to_csv('house_price_predictions.csv',index=False)
