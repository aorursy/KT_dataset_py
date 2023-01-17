import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno

import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_df
test_df
train_df.describe()
plt.figure(figsize=(18,18))
sns.heatmap(train_df.corr(),annot=True,cmap="Blues",fmt='.1f',square=True)
train_df.drop(['Id'],axis=1,inplace=True)
test_df.drop(['Id'],axis=1,inplace=True)
# View features that are highly correlated with SalePrice
corrs = train_df.corr()[['SalePrice']]
corrs = corrs[corrs['SalePrice']>0.5]
corrs = corrs.sort_values(by='SalePrice',ascending=False)

high_corr_feats = corrs.index[1:]

fig, axes = plt.subplots(5,2,figsize=(13,16))

for i, ax in enumerate(axes.flatten()):
    feat = high_corr_feats[i]
    sns.scatterplot(x=train_df[feat], y=train_df['SalePrice'], ax=ax)
    plt.xlabel(feat)
    plt.ylabel('Sale Price')
plt.tight_layout()
train_df.shape
# Drop GrLivArea outliers
train_df.drop(train_df[(train_df['SalePrice'] < 300000) & 
                       (train_df['GrLivArea'] > 4000)].index,
                       inplace=True)

# Drop TotalBsmtSF and 1stFlrSF outliers
train_df.drop(train_df[(train_df['TotalBsmtSF'] > 6000) | 
                       (train_df['1stFlrSF'] > 4000)].index,
                       inplace=True)
train_df.shape
fig, axes = plt.subplots(1,3,figsize=(14,4))
feats = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF']

for i, ax in enumerate(axes.flatten()):
    feat = feats[i]
    sns.scatterplot(x=train_df[feat], y=train_df['SalePrice'], ax=ax)
    plt.xlabel(feat)
    plt.ylabel('Sale Price')
    
plt.tight_layout()
df = pd.concat([train_df.drop(['SalePrice'],axis=1),
                test_df]).reset_index(drop=True)
df.shape
msno.matrix(train_df)
msno.matrix(test_df)
df_na = 100 * df.isnull().sum() / len(df)
df_na = pd.DataFrame(df_na,columns=['%NA'])
df_na = df_na.sort_values('%NA', ascending=False)
df_na = df_na[df_na['%NA']>0]

plt.figure(figsize=(14,6))
sns.barplot(x=df_na.index,y=df_na['%NA'],)
plt.xticks(rotation = '90')
plt.title('Feature Missing Value Percentage',fontsize=20,fontweight='bold')
def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(df)
# 'None' if NA
for i in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
         'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageType', 'GarageFinish',
         'GarageQual', 'GarageCond', 'MasVnrType', 'FireplaceQu', 'MSSubClass']:
    df[i] = df[i].fillna('None')


# 0 if NA
for i in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
          'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']:
    df[i] = df[i].fillna(0)

    
# Exterior1st, Exterior2nd - mode if NA
for i in ['Exterior1st', 'Exterior2nd', 'KitchenQual', 'Electrical', 'MSZoning',
         'SaleType', 'Functional']:
    df[i] = df[i].fillna(df[i].mode()[0])

    
# LotFrontage - Take median of neighborhood
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# Utilities - Drop, as all are 'AllPub', except one 'NoSeWa in training data.
df.drop(['Utilities'],inplace=True,axis=1)
missing_percentage(df)
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['OverallCond'] = df['OverallCond'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

var = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 
        'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 
        'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape',
        'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass',
        'OverallCond', 'YrSold', 'MoSold']

for i in var:
    mdl = LabelEncoder().fit(list(df[i].values))
    df[i] = mdl.transform(list(df[i].values))

df[var].head()
df['Total_SF_Main'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
#df['Total_Porch_SF'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
#df['Total_Bathrooms'] = df['BsmtFullBath'] + df['FullBath'] + 0.5*(df['HalfBath'] + df['BsmtHalfBath'])
#df['YrBltRemod'] = df['YearBuilt'] + df['YearRemodAdd']
#df['Total_sqr_footage'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF']
#df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
#df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
#df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
#df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
#df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
sns.distplot(train_df['SalePrice'])
mu = train_df['SalePrice'].mean()
med = train_df['SalePrice'].median()
std = train_df['SalePrice'].std()
skew = train_df['SalePrice'].skew()
kurt = train_df['SalePrice'].kurt()

print('SalePrice \n mean = {:.2f} \n median = {:.2f} \n standard deviation = {:.2f} \n skew = {:.2f} \n kurtosis = {:.2f}'.format(mu, med, std, skew, kurt))
stats.probplot(train_df['SalePrice'], plot=plt)
sns.residplot('GrLivArea', 'SalePrice', data=train_df)
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

mu = train_df['SalePrice'].mean()
med = train_df['SalePrice'].median()
std = train_df['SalePrice'].std()
skew = train_df['SalePrice'].skew()
kurt = train_df['SalePrice'].kurt()

print('SalePrice \n mean = {:.2f} \n median = {:.2f} \n standard deviation = {:.2f} \n skew = {:.2f} \n kurtosis = {:.2f}'.format(mu, med, std, skew, kurt))

sns.distplot(train_df['SalePrice'])
plt.figure()
stats.probplot(train_df['SalePrice'], plot=plt)
sns.residplot('GrLivArea', 'SalePrice', data=train_df)
numeric_var_skews = pd.DataFrame(df.dtypes[df.dtypes != 'object'].index,columns=['Numeric_Variables'])
numeric_var_skews['Skew'] = numeric_var_skews['Numeric_Variables'].apply(lambda x: df[x].skew())
numeric_var_skews.sort_values('Skew',ascending=False,inplace=True)
numeric_var_skews.reset_index(inplace=True,drop=True)
display(numeric_var_skews)
high_skew = numeric_var_skews[abs(numeric_var_skews['Skew']) > 0.75]

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew_vars = high_skew['Numeric_Variables']
for var in high_skew_vars:
    df[var] = boxcox1p(df[var], 0.15, #boxcox_normmax(df[var] + 1)
                      )
# Interestingly, not removing the first dummy variable actually improved
# the final test score, thus we keep drop_first=False. Normally, one 
# would want to remove one of the dummy variables to avoid collinearity
# in situations where the dummies represent all possible scenarios.
df_dummy = pd.get_dummies(df, #drop_first = True
                         )
df_dummy.shape
def overfit_reducer(df):
    """
    This function takes in a dataframe and returns a list of features that are overfitted.
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.94:
            overfit.append(i)
    overfit = list(overfit)
    return overfit

overfitted_features = overfit_reducer(df_dummy[:train_df.shape[0]])

df_dummy = df_dummy.drop(overfitted_features, axis=1)
# Remove additional outliers
train = df_dummy[:train_df.shape[0]]
Y_train = train_df['SalePrice'].values

import statsmodels.api as sm
ols = sm.OLS(endog = Y_train,
             exog = train)
fit = ols.fit()
test2 = fit.outlier_test()['bonf(p)']

outliers = list(test2[test2<1e-2].index)

print('There were {:.0f} outliers at indices:'.format(len(outliers)))
print(outliers)

train_df = train_df.drop(train_df.index[outliers])
df_dummy = df_dummy.drop(df_dummy.index[outliers])
df_dummy.shape
# Helpful imports
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR

# Designate preprocessed train and test data
train = df_dummy[:train_df.shape[0]]
test = df_dummy[train_df.shape[0]:]
Y_train = train_df['SalePrice'].values

# Cross validation strategy
def rmsle_cv(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, Y_train,
            scoring='neg_mean_squared_error', cv=kf))
    return(rmse)
models = pd.DataFrame([],columns=['model_name','model_object','score_mean','score_std'])
knr = KNeighborsRegressor(9, weights='distance')
score = rmsle_cv(knr)
print('KNN Regression score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['knr',knr,score.mean(),score.std()]
from sklearn.linear_model import SGDRegressor
sgd = make_pipeline(RobustScaler(), SGDRegressor(alpha=1000000000000000,l1_ratio=1))
score = rmsle_cv(sgd)
print('SGD score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['sgd',sgd,score.mean(),score.std()]
rfr = RandomForestRegressor()
score = rmsle_cv(rfr)
print('Random Forest score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['rfr',rfr,score.mean(),score.std()]
lnr = LinearRegression()
score = rmsle_cv(lnr)
print('Linear Regression score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['lnr',lnr,score.mean(),score.std()]
ridg = make_pipeline(RobustScaler(), Ridge(alpha = .17,normalize=True, random_state=4))
score = rmsle_cv(ridg)
print('Ridge score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['ridg',ridg,score.mean(),score.std()]
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.02, gamma=0.00046))
score = rmsle_cv(svr)
print('SVR score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['svr',svr,score.mean(),score.std()]
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.00042, max_iter=100000, random_state=1))
score = rmsle_cv(lasso)
print('Lasso Score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['lasso',lasso,score.mean(),score.std()]
e_net = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.00045, l1_ratio=0.9, random_state=1))
score = rmsle_cv(e_net)
print('Elastic Net score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['e_net',e_net,score.mean(),score.std()]
kr = make_pipeline(RobustScaler(), KernelRidge(alpha=0.04, kernel='polynomial', degree=1, coef0=2.5))
score = rmsle_cv(kr)
print('Kernel Ridge score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['kr',kr,score.mean(),score.std()]
dtr = make_pipeline(RobustScaler(), DecisionTreeRegressor(random_state=0, max_depth=20))
score = rmsle_cv(dtr)
print('Decision Tree score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['dtr',dtr,score.mean(),score.std()]
gbr = GradientBoostingRegressor(n_estimators=3000, 
            learning_rate=0.05, max_depth=4, max_features='sqrt',
            min_samples_leaf=1, min_samples_split=2, loss='huber',
            random_state=5,)
score = rmsle_cv(gbr)
print('Gradient Boosting score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['gbr',gbr,score.mean(),score.std()]
lgbr = lgb.LGBMRegressor(objective='regression',num_leaves=5,
        learning_rate=0.05, n_estimators=720, max_bin = 55,
        bagging_fraction = 0.8, bagging_freq = 5, 
        feature_fraction = 0.2319, feature_fraction_seed=9, bagging_seed=9,
        min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lgbr)
print('LightGBM score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['lgbr',lgbr,score.mean(),score.std()]
xgbr = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
        learning_rate=0.05, max_depth=3, min_child_weight=1.7817,
        n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571,
        subsample=0.5213, silent=True, random_state =7, nthread = -1)
score = rmsle_cv(xgbr)
print('XGBoost score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
models.loc[len(models)] = ['xgbr',xgbr,score.mean(),score.std()]
models.sort_values(by='score_mean',inplace=True)
models.reset_index(inplace=True,drop=True)
models
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)  
# First, we create a list of all model combinations
from itertools import combinations 

def subset(lst, count): 
    return list(set(combinations(lst, count)))

model_list = list(models[models['score_mean']<0.11]['model_name'])
combo = list()

for i in range(1,len(model_list)):
    combo = combo + subset(model_list, i)

print('There are {:.0f} combinations. First 20 include:'.format(len(combo)))
combo[:20]
# Now, we'll apply AveragingModels to every combination. Note, this may take a while.

# Commenting out this section for purposes of posting on Kaggle.

'''
model_scores = pd.DataFrame([],columns=['models_averaged','score_mean','score_std'])

for i in range(len(combo)):
    mods = list()
    for j in range(len(combo[i])):
        mods = mods + list(models[models['model_name']==list(combo[i])[j]]['model_object'])
    avg = AveragingModels(models = mods)
    score = rmsle_cv(avg)
    model_scores.loc[len(model_scores)] = [combo[i],score.mean(),score.std()]

model_scores = model_scores.sort_values(by='score_mean')
model_scores.head(25)
'''
simple_avg_final = AveragingModels(models = (lasso, gbr, lgbr, kr))
score = rmsle_cv(simple_avg_final)
print('Simple Average score = {:.4f}  (std = {:.4f})'.format(score.mean(), score.std()))
from mlxtend.regressor import StackingCVRegressor

stacked = StackingCVRegressor(regressors=(lasso, gbr, lgbr, kr),
                                meta_regressor=lasso,
                                use_features_in_secondary=True)

score = rmsle_cv(stacked)
print('Stacked score = {:.8f}  (std = {:.4f})'.format(score.mean(), score.std()))
class StackingCVRegressor_Scratch(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
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
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
stacked_scratch = StackingCVRegressor_Scratch(base_models = (lasso, gbr, lgbr, kr),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_scratch)
print('Stacked (Scratch) score = {:.8f}  (std = {:.4f})'.format(score.mean(), score.std()))
stacked_final = StackingCVRegressor(regressors=(svr, ridg, xgbr),
                                meta_regressor=e_net,
                                use_features_in_secondary=True)

score = rmsle_cv(stacked_final)
print('stacked_final score = {:.8f}  (std = {:.4f})'.format(score.mean(), score.std()))
model_1 = simple_avg_final
model_2 = stacked_final
mod_1_share = .5
mod_2_share = .5

model_1.fit(train.values, Y_train)
model_1_test_predictions = np.expm1(model_1.predict(test.values))

model_2.fit(train.values, Y_train)
model_2_test_predictions = np.expm1(model_2.predict(test.values))

test_predictions = mod_1_share * model_1_test_predictions + mod_2_share * model_2_test_predictions
test_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')[['Id']]
test_id['SalePrice'] = np.round(test_predictions,2)
test_id.to_csv('out51_simple(lasso,gbr,lgbr,kr)_meta(e_net,svr,ridg,xgbr).csv',index=False)