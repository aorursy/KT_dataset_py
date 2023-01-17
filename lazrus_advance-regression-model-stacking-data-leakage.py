# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as ms

from scipy import stats

from scipy.stats import norm, skew



#Data 

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

smp = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')



#Getting Info on the Data

#with open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt') as fhandle:

    #for line in fhandle.readlines():

        #print(line)
# Visualizing Outliers

plt.figure(figsize=(15,10))

plt.scatter(train.GrLivArea,train.SalePrice,c='orange',s=90,alpha=0.4)

plt.ylabel('Sales Price',fontsize=15)

plt.xlabel('GrLivArea',fontsize=15)

plt.title('Checking For Outliers',fontsize=15)

plt.grid( alpha=0.5,color='lightslategrey')

sp = plt.gca().spines

sp['top'].set_visible(False)

sp['right'].set_visible(False);
# Removing Outliers

train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index,inplace=True)

# Visualizing Outliers

plt.figure(figsize=(15,10))

plt.scatter(train.GrLivArea,train.SalePrice,c='orange',s=90,alpha=0.4)

plt.ylabel('Sales Price',fontsize=15)

plt.xlabel('GrLivArea',fontsize=15)

plt.title('Checking For Outliers',fontsize=15)

plt.grid( alpha=0.5,color='lightslategrey')

sp = plt.gca().spines

sp['top'].set_visible(False)

sp['right'].set_visible(False);
#Making Canvas

canv,axs = plt.subplots(2,2)

canv.set_size_inches(18,13)

canv.tight_layout(pad=7.0)

title = 'Before'



#Plotting and Tranforming



for rw in range(2):

    plt.sca(axs[rw][0])

    sns.distplot(train['SalePrice'] , fit=norm, ax = plt.gca())

    

    mu,sigma = norm.fit(train['SalePrice']) # Getting Fitting Parameters

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best',frameon=False)

    

    sp = plt.gca().spines

    sp['top'].set_visible(False)

    sp['right'].set_visible(False)

    plt.grid( alpha=0.5,color='lightslategrey')

    

    plt.ylabel('Frequency')

    plt.title('SalePrice Distribution {} Tranformation'.format(title))

    

    plt.sca(axs[rw][1])

    

    stats.probplot(train['SalePrice'], plot=plt)

    plt.title('Probability Plot {} Tranformation'.format(title))

    sp = plt.gca().spines

    sp['top'].set_visible(False)

    sp['right'].set_visible(False)

    

    if rw != 0: # Little bit of automation is not bad right!

        break

    

    train["SalePrice"] = np.log1p(train["SalePrice"])

    title = 'After'
#Train Data

train_na = (train.isnull().sum() / len(train)) * 100

train_na = train_na.drop(train_na[train_na==0].index).sort_values(ascending=False)



train_na_df = pd.DataFrame({'Missing Ratio' :train_na})

train_na_df
#Test Data

test_na = (test.isnull().sum() / len(test)) * 100

test_na = test_na.drop(test_na[test_na==0].index).sort_values(ascending=False)



test_na_df = pd.DataFrame({'Missing Ratio' :test_na})

test_na_df
data = train_na

title = 'Train'

for _ in range(2):

    plt.figure(figsize=(18, 10))

    sns.barplot(x=data.index, y=data)

    plt.xticks(rotation='90')  

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature in {} Data'.format(title), fontsize=15)

    

    sp = plt.gca().spines

    sp['top'].set_visible(False)

    sp['right'].set_visible(False)

    

    if title != 'Train': # Little bit of automation is never bad!

        break

    data = test_na

    title = 'Test'
train['PoolQC'] = train['PoolQC'].fillna('None')

test['PoolQC'] = test['PoolQC'].fillna('None')
train["MiscFeature"] = train["MiscFeature"].fillna("None")

test["MiscFeature"] = test["MiscFeature"].fillna("None")
train["Alley"] = train["Alley"].fillna("None")

test["Alley"] = test["Alley"].fillna("None")
train["Fence"] = train["Fence"].fillna("None")

test["Fence"] = test["Fence"].fillna("None")
train["FireplaceQu"] = train["FireplaceQu"].fillna("None")

test["FireplaceQu"] = test["FireplaceQu"].fillna("None")
mapper = train.groupby("Neighborhood").median()['LotFrontage'].to_dict()



for k,v in mapper.items():

    train.loc[(train['LotFrontage'].isnull() == True) & (train['Neighborhood'] == k), 'LotFrontage'] = v

    test.loc[(test['LotFrontage'].isnull() == True) & (test['Neighborhood'] == k), 'LotFrontage'] = v
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    train[col] = train[col].fillna('None')

    test[col] = test[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    train[col] = train[col].fillna(0)

    test[col] = test[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    train[col] = train[col].fillna(0)

    test[col] = test[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    train[col] = train[col].fillna('None')

    test[col] = test[col].fillna('None')
#MasVnrType

train["MasVnrType"] = train["MasVnrType"].fillna("None")

test["MasVnrType"] = test["MasVnrType"].fillna("None")



#MasVnrArea

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)

test["MasVnrArea"] = test["MasVnrArea"].fillna(0)
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode().item())

test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode().item())
train.drop('Utilities',axis=1,inplace=True)

test.drop('Utilities',axis=1,inplace=True)
train["Functional"] = train["Functional"].fillna("Typ")

test["Functional"] = test["Functional"].fillna("Typ")
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
for col in ['Exterior1st','Exterior2nd']:

    train[col] = train[col].fillna(train[col].mode()[0])

    test[col] = test[col].fillna(test[col].mode()[0])
train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])

test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
train['MSSubClass'] = train['MSSubClass'].fillna("None")

test['MSSubClass'] = test['MSSubClass'].fillna("None")
train_na = (train.isnull().sum() / len(train)) * 100

train_na = train_na.drop(train_na[train_na==0].index).sort_values(ascending=False)



train_na_df = pd.DataFrame({'Missing Ratio' :train_na})

train_na_df
test_na = (test.isnull().sum() / len(test)) * 100

test_na = test_na.drop(test_na[test_na==0].index).sort_values(ascending=False)



test_na_df = pd.DataFrame({'Missing Ratio' :test_na})

test_na_df
#MSSubClass=The building class

train['MSSubClass'] = train['MSSubClass'].apply(str)

test['MSSubClass'] = test['MSSubClass'].apply(str)



#Changing OverallCond into a categorical variable

train['OverallCond'] = train['OverallCond'].astype(str)

test['OverallCond'] = test['OverallCond'].astype(str)



#Year and month sold are transformed into categorical features.

train['YrSold'] = train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)



test['YrSold'] = test['YrSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(set(train[c].unique().tolist() + test[c].unique().tolist())))

    train[c] = lbl.transform(list(train[c].values))

    test[c] = lbl.transform(list(test[c].values))
# Adding total sqfootage feature 

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
#Seperating Columns for Skew check

y_train = train.SalePrice

train.drop(['SalePrice','Id'],axis=1,inplace=True)

test_Ids = test['Id']

test.drop('Id',axis=1,inplace=True)
numeric_f = train.dtypes[train.dtypes != "object"].index



# Check the skew of all numerical features



skewed_f = train[numeric_f].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew in train data' :skewed_f})

skewness.head(10)
#Transforming train Data



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features in train data to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    train[feat] = boxcox1p(train[feat], lam)
numeric_f = test.dtypes[test.dtypes != "object"].index



# Check the skew of all numerical features



skewed_f = test[numeric_f].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew in test data' :skewed_f})

skewness.head(10)
#Transforming test Data



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features in test data to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    test[feat] = boxcox1p(test[feat], lam)
#Getting Dummy Variables

train = pd.get_dummies(train)

test = pd.get_dummies(test)



#Balancing Data Sets

missing_cols = set(train.columns) - set(test.columns)

for c in missing_cols:

    test[c] = 0  



missing_cols = set(test.columns) - set(train.columns)

for c in missing_cols:

    train[c] = 0

    

test = test[train.columns.tolist()]



#Checking Shapes

train.shape,test.shape
#Importing Libraries



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Define a cross Validation Strategy



n_folds = 5

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, verbosity=0,

                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
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
averaged_score = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_score)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

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

                instance.fit(X[train_index], y.iloc[train_index])

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

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#rmsle Func

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.70 +

               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
Predictions = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
subm = pd.DataFrame()

subm['Id'] = test_Ids

subm['SalePrice'] = Predictions

subm.to_csv('submission.csv',index=False)