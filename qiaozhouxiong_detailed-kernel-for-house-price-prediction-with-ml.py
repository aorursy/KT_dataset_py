import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from scipy import stats

from scipy.stats import norm

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

print("libraries imported sucessfully")
train_data = pd.read_csv('../input/train.csv')

val_data = pd.read_csv('../input/test.csv')

train_data.head()
if len(set(train_data.Id)) == train_data.shape[0]:

    print("There is no repeated data in training data\n")

else:

    print("There is no repeated data in training data\n")

    

if len(set(val_data.Id)) == val_data.shape[0]:

    print("There is no repeated data in testing data")

else:

    print("There is no repeated data in testing data")
n_train = train_data.shape[0]

n_test = val_data.shape[0]

print('\n No. of training data: {}\n No. of testing data: {}\n'.format(n_train, n_test))
print('the number of category features:', len(train_data.select_dtypes(include = ['object']).columns))

print('the number of numerical features:', len(train_data.select_dtypes(exclude = ['object']).columns))
fig, axes = plt.subplots(1,2, figsize=(16, 5))

sns.distplot(train_data[["SalePrice"]], ax = axes[0])

## get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_data["SalePrice"])

print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

axes[0].legend(["norm dist. ($\mu=${:.2f} and $\sigma=${:.2f})".format(mu, sigma)], loc='best')

axes[0].set_xlabel('Sale price')

axes[0].set_ylabel('Normalized Frequency')

axes[0].set_title('sale price histogram')

stats.probplot(train_data["SalePrice"], plot = axes[1])
train_data[["SalePrice"]] = np.log(train_data["SalePrice"])

fig, axes = plt.subplots(1,2, figsize=(16, 5))

sns.distplot(train_data[["SalePrice"]], ax = axes[0])

## get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_data["SalePrice"])

print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

axes[0].legend(["norm dist. ($\mu=${:.2f} and $\sigma=${:.2f})".format(mu, sigma)], loc='best')

axes[0].set_xlabel('Sale price')

axes[0].set_ylabel('Normalized Frequency')

axes[0].set_title('sale price histogram')

stats.probplot(train_data["SalePrice"], plot = axes[1])
nan_percent = pd.DataFrame(train_data.isna().sum()/1460*100).reset_index()

nan_percent.columns = ["category", "NAN_Percent"]

nan_data = nan_percent[nan_percent.NAN_Percent>0].sort_values(by = ["NAN_Percent"], ascending = True)

fig, ax = plt.subplots(figsize=(20,5))

nan_data.set_index("category").plot.barh(color ='b',ax = ax)
df_all = pd.concat([train_data.iloc[:,:-1], val_data], axis = 0)

y_train = train_data.iloc[:,-1]
df_all[nan_data.category].head(1)
## just replace it with "none"

df_all["PoolQC"].fillna("None", inplace=True)

print('No. of nan in PoolQC: {: }'.format(df_all["PoolQC"].isna().sum()))
df_all.replace({"PoolQC":{"Ex":4, "Gd":3, "TA":2, "Fa":1, "None":0}}, inplace=True)
print('PoolQC types:{}'.format(df_all["PoolQC"].dtypes))
df_all["MiscFeature"].fillna("MNA", inplace=True)

print('No. of nan in Miscfeatures: {: }'.format(df_all["MiscFeature"].isna().sum()))
df_all["Alley"].fillna("None", inplace=True)
print('No. of nan in Alley: {: }'.format(df_all["Alley"].isna().sum()))
df_all.replace({"Alley": {"Gravel":1, "Paved":2, "None":0}}, inplace=True)
df_all["Fence"].fillna("None", inplace=True)
df_all.replace({"Fence": {"GdPrv":4, "MnPrv":3, "GdWo":2, "MnWw":1, "None":0}}, inplace=True)
df_all.replace({"FireplaceQu": {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.nan:0}}, inplace=True)
df_all[["LotFrontage"]].describe().T
df_all["LotFrontage"] = df_all.groupby('Neighborhood')["LotFrontage"].transform(lambda x: x.fillna(x.median()))
print('No. of nan in logFrontage: {: }'.format(df_all["LotFrontage"].isna().sum()))
df_all["GarageType"].fillna("NGag", inplace=True)

df_all["GarageYrBlt"].fillna(0, inplace=True)

for col in ["GarageCars", "GarageArea"]:

    df_all[col].fillna(0, inplace=True)
df_all.replace({"GarageFinish":{"Fin":3, "RFn":2, "Unf":1, np.nan:0}}, inplace=True)

df_all.replace({"GarageQual":{"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.nan:0}}, inplace=True)

df_all.replace({"GarageCond":{"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.nan:0}}, inplace=True)
df_all.replace({"BsmtQual":{"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.nan:0}}, inplace=True)

df_all.replace({"BsmtCond":{"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1, np.nan:0}}, inplace=True)

df_all.replace({"BsmtExposure":{"Gd":4, "Av":3, "Mn":2, "No":1, np.nan:0}}, inplace=True)

df_all.replace({"BsmtFinType1":{"GLQ":6, "ALQ":5,"BLQ":4, "Rec":3, "LwQ":2, "Unf":1, np.nan:0}}, inplace=True)

df_all.replace({"BsmtFinType2":{"GLQ":6, "ALQ":5,"BLQ":4, "Rec":3, "LwQ":2, "Unf":1, np.nan:0}}, inplace=True)
df_all["MasVnrType"].fillna('None', inplace=True)

df_all["MasVnrArea"].fillna(0, inplace=True)
df_all["Electrical"].fillna("SBrkr", inplace=True)
nan_percent = pd.DataFrame(df_all.isna().sum()).reset_index()

nan_percent.columns = ["category", "NAN_Percent"]

nan_data = nan_percent[nan_percent.NAN_Percent>0].sort_values(by = ["NAN_Percent"], ascending = True)

fig, ax = plt.subplots(figsize=(20,5))

nan_data.set_index("category").plot.barh(color ='b',ax = ax)
df_all['MSZoning'] = df_all.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

df_all['Functional'].fillna('Typ', inplace=True)

df_all["BsmtHalfBath"].fillna(0, inplace=True)

df_all["BsmtFullBath"].fillna(0, inplace=True)

df_all["SaleType"].fillna("WD", inplace=True)
df_all.fillna(method="pad", inplace=True)
# Some of the non-numeric predictors are stored as numbers; convert them into strings

df_all['MSSubClass'] = df_all['MSSubClass'].apply(str)

df_all['MoSold'] = df_all['MoSold'].astype(str)
cat_features = df_all.select_dtypes(include = ["object"]).columns

num_features = df_all.select_dtypes(exclude = ["object"]).columns

print("number of categorica features: {}".format(len(cat_features)))

df_all.drop(columns=['Id'], axis=1, inplace=True)
import copy

df_all_copy = copy.deepcopy(df_all)
df_allNum = pd.get_dummies(df_all)

df_allNum.shape
df_allNum.head(2)
df_corr = pd.concat([df_all.loc[:n_train-1, num_features], train_data["SalePrice"]], axis=1)

corrmap = df_corr.corr()

fig, ax = plt.subplots(figsize=(15,10))

sns.heatmap(corrmap, ax=ax)

main_features = corrmap["SalePrice"].sort_values(ascending=False).head(20).index

corrmap["SalePrice"].sort_values(ascending=False).head(20)
fig, ax = plt.subplots(5,2, figsize = (20,10))

plt.subplots_adjust(wspace=0.2, hspace=0.5)

for ind, col in enumerate(main_features[1:11]):

    cl = int(np.floor(ind/5))

    rw = np.mod(ind,5)

    ax[rw][cl].plot(df_all.loc[:n_train-1, [col]], train_data["SalePrice"], '.')

    ax[rw][cl].set_xlabel(col)

    ax[rw][cl].set_ylabel("SalePrice")
## We add some square order of the main features to the data

for col in main_features[1:11]:

    col2d = col+'-2d'

    df_allNum[[col2d]] = df_allNum[[col]]**2     

    col3d = col+'-3d'

    df_allNum[[col3d]] = df_allNum[[col]]**3  

print("size of the df_allNum: {}".format(df_allNum.shape))
fig, axes = plt.subplots(5,2, figsize = (20,10))

for ind, col in enumerate(main_features[1:11]):

    cl = int(np.floor(ind/5))

    rw = np.mod(ind,5)   

    sns.distplot(df_all[col], ax = axes[rw][cl])
num_features = df_allNum.select_dtypes(exclude=['object']).columns

skew_feats = df_allNum[num_features].apply(lambda x: abs(stats.skew(x))).sort_values(ascending = False)

skewness = pd.DataFrame({"Skewness":skew_feats})

skewness.head(20).T
from scipy.special import boxcox1p

skew_feats = df_allNum[num_features].apply(lambda x: abs(stats.skew(x))).sort_values(ascending = False)

skewness = pd.DataFrame({"Skewness":skew_feats})

skewed_features = skewness[abs(skewness)>0.5].index

lam = 0.15

for feat in skewed_features:

    df_allNum[feat] = boxcox1p(df_allNum[feat], lam)
skew_feats = df_allNum[num_features].apply(lambda x: abs(stats.skew(x))).sort_values(ascending = False)

skewness = pd.DataFrame({"Skewness":skew_feats})

skewness.head(20).T
fig, axes = plt.subplots(5,2, figsize=(16, 5))

plt.subplots_adjust(wspace=0.1, hspace=1)

for ind, feat in enumerate(skewness.head(10).index):

    cl = int(np.floor(ind/5))

    rw = np.mod(ind,5)

    sns.distplot(df_allNum[feat], ax = axes[rw][cl])
#df_allNum.drop(skewness.head(20).index[10:], axis=1, inplace=True)

df_allNum.drop(["Id"], axis=1, inplace=True)
#!pip install xgboost

#!pip install lightgbm
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV,BayesianRidge, LassoLarsIC, LinearRegression, SGDRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale, RobustScaler, PolynomialFeatures

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
## validation function

n_folds = 20;

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle = True, random_state = 42)

    rmse = np.sqrt(-cross_val_score(model, df_allNum[:n_train], y_train, scoring = "neg_mean_squared_error", cv = kf))

    return(rmse)
linearReg = LinearRegression()

score = rmsle_cv(linearReg)

print("\n LinearRegression score:{:.4f} ({:.4f}) \n".format(score.mean(), score.std()))
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=[1, 0.1, 0.01, 0.001, 0.0005, 0.0001], random_state=1))

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNetCV(alphas=[1, 0.1, 0.01, 0.001, 0.0005, 0.0001], random_state=1))

score = rmsle_cv(ENet)

print("\n ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score = rmsle_cv(GBoost)

print("\n GBost score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.04, max_depth=4, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =2, nthread = -1)

score = rmsle_cv(model_xgb)

print("\n model_xgb score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=6,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(model_lgb)

print("\n ligth GBM score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
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
averaged_models = AveragingModels(models = (lasso, ENet, model_xgb))

score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
model_xgb.fit(df_allNum[:n_train], y_train)

xgb_train_pred = model_xgb.predict(df_allNum[:n_train])

print(rmsle(y_train, xgb_train_pred))

xgb_pred = np.expm1(model_xgb.predict(df_allNum[n_train:]))
lasso.fit(df_allNum[:n_train], y_train)

las_train_pred = lasso.predict(df_allNum[:n_train])

print(rmsle(y_train, las_train_pred))

las_pred = np.expm1(lasso.predict(df_allNum[n_train:]))
model_lgb.fit(df_allNum[:n_train], y_train)

lgb_train_pred = model_lgb.predict(df_allNum[:n_train])

print(rmsle(y_train, lgb_train_pred))

lgb_pred = np.expm1(model_lgb.predict(df_allNum[n_train:]))
averaged_models.fit(df_allNum[:n_train], y_train)

avg_train_pred = averaged_models.predict(df_allNum[:n_train])

print(rmsle(y_train, avg_train_pred))

avg_pred = np.expm1(averaged_models.predict(df_allNum[n_train:]))
final_pred = 0.3*xgb_pred+0.3*las_pred+0.3*lgb_pred+0.1*avg_pred
sub = pd.DataFrame()

sub['Id'] = val_data.Id

sub['SalePrice'] = final_pred

sub.to_csv('submission_2.csv',index=False)
df_all["TotalHouseQuality1"] = df_all['OverallQual'] + df_all['OverallCond']

df_all["TotalHouseQuality2"] = df_all['OverallQual'] * df_all['OverallCond']

df_all['TotalSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']

df_all['YearsSinceRemodel'] = df_all['YrSold'].astype(int) - df_all['YearRemodAdd'].astype(int)

df_all['YearsSinceBuilt'] = df_all['YrSold'].astype(int) - df_all['YearBuilt'].astype(int)
df_all['BsmtVal1'] = df_all['TotalBsmtSF']*df_all['BsmtCond']

df_all['BsmtFins'] = df_all['BsmtFinSF1']+df_all['BsmtFinSF2']

df_all['BsmtFinsVal'] = df_all['BsmtFinType1']*df_all['BsmtFinSF1']+df_all['BsmtFinType2']*df_all['BsmtFinSF2']
df_all['TolBath'] = df_all['BsmtFullBath']+df_all['BsmtHalfBath']+df_all['FullBath']+df_all['HalfBath']

#other measurement way

df_all['TolBath2'] = df_all['BsmtFullBath']+0.5*df_all['BsmtHalfBath']+df_all['FullBath']+0.5*df_all['HalfBath']
df_all['GaragOverall1'] = df_all['GarageQual']*df_all['GarageCond']

df_all['GaragOverall1'] = df_all['GarageQual']+df_all['GarageCond']

df_all['GaragOverall2'] = df_all['GarageQual']**2+df_all['GarageCond']**2

df_all['GarageAge'] = df_all['YrSold'].astype(int)-df_all['GarageYrBlt']
df_all['haspool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_all['has2ndfloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['hasgarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_all['hasbsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_all['hasfireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df_all.select_dtypes(include=['object']).columns
print('YearRemodAdd data type:',df_all['YearRemodAdd'].dtypes)

print('YrSold data type:',df_all['YrSold'].dtypes)

print('YearBuilt data type:',df_all['YearBuilt'].dtypes)

print('GarageYrBlt data type:',df_all['GarageYrBlt'].dtypes)
sns.distplot(df_all['YrSold'].astype(int))
num_features = df_all.select_dtypes(exclude=['object']).columns

df_corr = pd.concat([df_all.loc[:n_train-1, num_features], train_data["SalePrice"]], axis=1)

corrmap = df_corr.corr()

fig, ax = plt.subplots(figsize=(15,10))

sns.heatmap(corrmap, ax=ax)

main_features = corrmap["SalePrice"].sort_values(ascending=False).head(20).index

corrmap["SalePrice"].sort_values(ascending=False).head(20).T
##creating another copy

df_all_copy = copy.deepcopy(df_all)
df_all = df_all_copy
df_allNum = pd.get_dummies(df_all)

print("The size of the training data:", (df_allNum.shape))
num_features = df_all.select_dtypes(exclude=['object']).columns

skew_feats = df_allNum[num_features].apply(lambda x: abs(stats.skew(x))).sort_values(ascending = False)

skewness = pd.DataFrame({"Skewness":skew_feats})

skewness.sort_values(['Skewness'],ascending=False).head(20).T
## We add some square order of the main features to the data

for col in main_features[1:]:

    col2d = col+'-2d'

    df_allNum[[col2d]] = df_allNum[[col]]**2  

    col3d = col+'-3d'

    df_allNum[[col3d]] = df_allNum[[col]]**3  

    

print("size of the df_allNum: {}".format(df_allNum.shape))
from scipy.special import boxcox1p

skew_feats = df_allNum[num_features].apply(lambda x: abs(stats.skew(x))).sort_values(ascending = False)

skewness = pd.DataFrame({"Skewness":skew_feats})

skewed_features = skewness[abs(skewness)>0.5].index

lam = 0.15

for feat in skewed_features:

    df_allNum[feat] = boxcox1p(df_allNum[feat], lam)
skew_feats = df_allNum[num_features].apply(lambda x: abs(stats.skew(x))).sort_values(ascending = False)

skewness = pd.DataFrame({"Skewness":skew_feats})

skewness.sort_values(["Skewness"],ascending=False).head(20).T
linearReg = LinearRegression()

score = rmsle_cv(linearReg)

print("\n LinearRegression score:{:.4f} ({:.4f}) \n".format(score.mean(), score.std()))
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=[1, 0.1, 0.01, 0.001, 0.0005, 0.0001], random_state=1))

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.model_selection import GridSearchCV
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.01, max_depth=4, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =2, nthread = -1)



score = rmsle_cv(model_xgb)

print("model_xgb estimator score:{:.4f} ({:.4f})".format(score.mean(), score.std()))
param_test1 = {

    'max_depth':range(3,8,1),

    'min_child_weight':np.arange(1,9,1),

}

xgb_gr = GridSearchCV(estimator=model_xgb, param_grid=param_test1, scoring='r2', cv=5)

xgb_gr.fit(df_allNum[:n_train], y_train)

print("Best score: %0.3f" % xgb_gr.best_score_)

score = rmsle_cv(xgb_gr.best_estimator_)

print("model_xgb estimator score:{:.4f} ({:.4f})".format(score.mean(), score.std()))
import torch

torch.cuda.is_available()
!conda install pytorch torchvision cudatoolkit=9.0 -c pytorch