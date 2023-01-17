# Data Processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold,cross_validate, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

# Data Visualizing
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Data Modeling
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, LassoLarsIC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# Data Evalutation
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve

# Math
import math
from scipy.stats import norm
from scipy import stats

# Warning Removal
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
# Let's import and put the train and test datasets in  pandas dataframe
train = pd.read_csv('../input/house-prices-data/train.csv')
test = pd.read_csv('../input/house-prices-data/test.csv')

# Drop the 'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Concatenate both train and test dataset
all_data = pd.concat((train, test)).reset_index(drop=True)
# Check the overall insight of the dataset
all_data.describe()
# Check to see Null values and Data Type of each feature
all_data.info(verbose=True)
# Target is continuous variable, using correlation to know which features may help prediction
all_data.corr()['SalePrice'].sort_values(ascending = False)
fig = plt.figure()
sns.scatterplot(x=train['OverallQual'], y=train['SalePrice'])

fig = plt.figure()
sns.boxplot(x=train['OverallQual'], y=train['SalePrice'])

fig = plt.figure()
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])
# Remove outliers in dataset
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)

# df[(df['GrLivArea']>4500) & (df['SalePrice']<300000)]
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'], hue=train['OverallQual'])
sns.scatterplot(x=train['ExterQual'], y=train['SalePrice'])

fig = plt.figure()
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'], hue=train['ExterQual'])
sns.distplot(train['SalePrice'])

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#We use the numpy fuction log1p which applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

# plot the histogram.
sns.distplot(train['SalePrice'], hist=True, kde=True, fit=norm, color='#e74c3c')

(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
zero_count = (train.isnull()).sum().sort_values(ascending=False) # (df == 0).sum() # 
zero_count_df = pd.DataFrame(zero_count)
zero_count_df.drop('SalePrice', axis=0, inplace=True)
zero_count_df.columns = ['count_0']

# https://stackoverflow.com/questions/31859285/rotate-tick-labels-for-seaborn-barplot/60530167#60530167
sns.set(style='whitegrid')
plt.figure(figsize=(13,8))
sns.barplot(x=zero_count_df.index, y=zero_count_df['count_0'])
plt.xticks(rotation=90)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train['SalePrice']
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
# MSSubClass: missing value means No Building class
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# The size and area of individual house in the same neighborhood tend to be similar,
# thus filling missing value with the median neighborhood LotFronttage 
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))

# Alley: NA means No Alley
all_data['Alley'].fillna('None', inplace=True)

# MasVnrType,MasVnrArea: NA means no masonry veneer for these houses
all_data['MasVnrType'].fillna('None', inplace=True)
all_data['MasVnrArea'] = all_data.groupby('MasVnrType')['MasVnrArea'].apply(lambda x: x.fillna(x.median()))

# BsmtQual,BsmtCond,BsmtExposure: 
# Using Tableau Prep or visualization to check their distribution and fill with most common values.
all_data['BsmtQual'].fillna('TA', inplace=True)
all_data['BsmtCond'].fillna('TA', inplace=True)
all_data['BsmtExposure'].fillna('No', inplace=True)

# BsmtFinType1, BsmtFinType2:
# Using Tableau Prep or visualization to check their distribution and fill with most common values.
# BsmtFinSF1,BsmtFinSF2,BsmtUnfSF:
# Using groupby on BsmtFinSF1 and BsmtFinSF2 to fill nan with median value.
# BsmtUnfSF: means either having no Bsmt or being built. Therefor fill nan with either None or median
all_data['BsmtFinType1'].fillna('Rec', inplace=True)
all_data['BsmtFinSF1'] = all_data.groupby('BsmtFinType1')['BsmtFinSF1'].apply(lambda x: x.fillna(x.median()))
all_data['BsmtFinType2'].fillna('Rec', inplace=True)
all_data['BsmtFinSF2'] = all_data.groupby('BsmtFinType2')['BsmtFinSF2'].apply(lambda x: x.fillna(x.median()))
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(all_data['BsmtUnfSF'].median())

# There is a certain correlation between GrLivArea and TotalBsmtSF. 
# Taking all non-value of GrLivArea and TotalBsmtSF into equation: average of GrLivArea/TotalBsmtSF
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(all_data['GrLivArea']/1.5)

# Electrical: fill nan with most common value
all_data['Electrical'].fillna(all_data['Electrical'].mode()[0], inplace=True)

# BsmtFullBath,BsmtHalfBath: fill nan with most common value
all_data['BsmtFullBath'].fillna(all_data['BsmtFullBath'].mode()[0], inplace=True)
all_data['BsmtHalfBath'].fillna(all_data['BsmtHalfBath'].mode()[0], inplace=True)

# Functional: fill nan with most common value
all_data['Functional'].fillna(all_data['Functional'].mode()[0], inplace=True)

# FireplaceQu: fill nan with most common value
all_data['FireplaceQu'].fillna('None', inplace=True)

# GarageType: fill nan with None, according to description
all_data['GarageType'].fillna('No', inplace=True)
# GarageYrBlt: Garage is either built in the same year with the house or no Garage at all.
all_data['GarageYrBlt'].fillna(all_data['YearBuilt'], inplace=True)

# GarageFinish: fill nan with no for not having finish
# GarageCars, GarageArea: fill nan with median value
# GarageQual, GarageCond: fill nan based on OveralQual distribution
all_data['GarageFinish'].fillna('No', inplace=True)
all_data['GarageCars'].fillna(all_data['GarageCars'].median(), inplace=True)
all_data['GarageArea'].fillna(all_data['GarageArea'].median(), inplace=True)
all_data['GarageQual'].fillna('TA', inplace=True)
all_data['GarageCond'].fillna('TA', inplace=True)

# PoolQC,Fence,MiscFeature: fill nan with None 
all_data['PoolQC'].fillna('None', inplace=True)
all_data['Fence'].fillna('None', inplace=True)
all_data['MiscFeature'].fillna('None', inplace=True)

# Exterior1st,Exterior2nd,SaleType: fill nan with the most common value
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'].fillna(all_data['SaleType'].mode()[0], inplace=True)

# Utilities: Having the majority of AllPub, and a few of NoSeWa. Thus removing is safe
all_data = all_data.drop(['Utilities'], axis=1)
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
# Getting dummies categorical features
all_data = pd.get_dummies(all_data)
print(all_data.shape)
train = all_data[:ntrain]
test = all_data[ntrain:]
# Pipeline for Scaler + Model: https://stackoverflow.com/questions/43366561/use-sklearns-gridsearchcv-with-a-pipeline-preprocessing-just-once
# Metrics in GridSearchCV: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

def Best_Parameters_model(est, para):
    model_table = {}
    
    MLA_name = est.__class__.__name__
    model_table['Model Name'] = MLA_name

    pipe = make_pipeline(RobustScaler(), GridSearchCV(estimator=est,
                                              param_grid=para,
                                              scoring='neg_root_mean_squared_error',
                                              cv=10,
                                              verbose=0, refit=True))
    pipe_result = pipe.fit(train, y_train)

    model_table['Best Test Accuracy Mean'] = pipe_result[1].best_score_
    model_table['Best Parameters'] = pipe_result[1].best_params_
        
    return model_table

# ridge = Ridge(random_state=0)
# ridge_para = {'alpha':[0.9, 0.5, 0.1, 0.01, 0.001]}
# Best_Parameters_model(ridge, ridge_para)

# ridge = Ridge(random_state=0)
# ridge_para = {'alpha':[0.9, 0.5, 0.1, 0.01, 0.001]}
# Best_Parameters_model(ridge, ridge_para)

# ENet = ElasticNet(random_state=0)
# ENet_para = {'alpha': [0.1, 0.01, 0.0001],
#             'l1_ratio': [0.9, 0.5, 0.1]}
# Best_Parameters_model(ENet, ENet_para)
def rmsle_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse.mean())
lasso = Lasso(alpha=0.0008,random_state=0)
rmsle_cv(lasso)
ridge = Ridge(alpha=15,random_state=0)
rmsle_cv(ridge)
ENet = ElasticNet(alpha=0.01, l1_ratio=0.05,random_state=0)
rmsle_cv(ENet)
lasso_coeficient = Lasso(alpha=0.0008,random_state=0).fit(train, y_train).coef_
print("Lasso picked " + str(sum(lasso_coeficient != 0)) + " variables and eliminated the other " +  str(sum(lasso_coeficient == 0)) + " variables")

ridge_coeficient = Ridge(alpha=15,random_state=0).fit(train, y_train).coef_
print("Lasso picked " + str(sum(ridge_coeficient != 0)) + " variables and eliminated the other " +  str(sum(ridge_coeficient == 0)) + " variables")

ENet_coeficient = ElasticNet(alpha=0.01, l1_ratio=0.05,random_state=0).fit(train, y_train).coef_
print("Lasso picked " + str(sum(ENet_coeficient != 0)) + " variables and eliminated the other " +  str(sum(ENet_coeficient == 0)) + " variables")
# Tuning hyperparameter: kaggle.com/duonghoanvu1/houseprice/edit
GBoost = GradientBoostingRegressor(criterion='friedman_mse',
                                   learning_rate=0.01,
                                   loss= 'huber',
                                   max_depth=4,
                                   max_features='sqrt',
                                   min_samples_split=5,
                                   n_estimators=4000,
                                   subsample=0.9,
                                   random_state = 0)
rmsle_cv(GBoost)
# Tuning hyperparameter: kaggle.com/duonghoanvu1/houseprice1/edit
XGBoost = XGBRegressor(n_estimators=2500,
                     max_depth=5,
                     num_leaves=30,
                     learning_rate=0.01,
                     booster='gbtree',
                     objective='reg:squarederror',
                     subsample=0.7,
                     colsample_bytree=0.8,
                     gamma=0.001,
                     random_state=0, 
                     n_jobs=-1)
rmsle_cv(XGBoost)
# Tuning hyperparameter: kaggle.com/duonghoanvu1/houseprice2/edit
LGBM = LGBMRegressor(n_estimators=2500,
                     max_depth=5,
                     num_leaves=30,
                     learning_rate=0.01,
                     boosting_type='gbdt',
                     objective='regression',
                     subsample=0.55,
                     colsample_bytree=0.70,
                     reg_alpha=0.01,
                     random_state=0, 
                     n_jobs=-1)
rmsle_cv(LGBM)
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
averaged_models = AveragingModels(models = (ridge, lasso, ENet, GBoost))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f}\n".format(score))
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
averaged_models.fit(train.values, y_train)
averaged_models_pred = averaged_models.predict(train.values)
print(rmsle(y_train, averaged_models_pred))
XGBoost.fit(train, y_train)
xgb_pred = XGBoost.predict(train)
#xgb_pred = np.expm1(xgb_pred)
print(rmsle(y_train, xgb_pred))
##### Generate average_model
LGBM.fit(train, y_train)
LGBM_pred = LGBM.predict(train)
print(rmsle(y_train, LGBM_pred))
ensemble = averaged_models_pred*0.70 + xgb_pred*0.15 + LGBM_pred*0.15
print(rmsle(y_train, ensemble))