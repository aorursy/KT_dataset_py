# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.style as style

from scipy.stats import skew









#Model Building



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



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(train.keys())
train.head()
train.tail()
sns.set_style("white")

sns.set_color_codes(palette='deep')

f,ax = plt.subplots(figsize=(8,7))



sns.distplot(train['SalePrice'], color ='b')

ax.set(ylabel = "Frequency")

ax.set(xlabel = "SalePrice")

ax.set(title = "SalePrice Distribution" )

sns.despine(trim=True, left=True)

plt.show()
# Skewness and Kurtosis



print("Skewness : %f" % train['SalePrice'].skew() )

print("Kurtosis : %f" % train['SalePrice'].kurt())
# In this case, We use the numpy fuction log1p which  applies log(1+x) to all elements of the column



train['SalePrice'] = np.log1p(train['SalePrice'])





# Check the Skewness



sns.distplot(train['SalePrice'], color = 'b')

ax.set(ylabel = "Frequency")

ax.set(xlabel = "SalePrice")

ax.set(title = "SalePrice Distribution")

sns.despine(trim=True, left=True)

plt.show()
plt.figure(figsize=(15,8))

sns.boxplot(x= train.YearBuilt, y= train.SalePrice)
fig, ax = plt.subplots(figsize=(16,8))

ax.scatter(x = train.GrLivArea, y= train.SalePrice )

ax.set_xlabel('GrLivArea')

ax.set_ylabel('SalePrice')

plt.show()
train.iloc[np.where(train.GrLivArea > 4000)]
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index,inplace=True)
fig, ax = plt.subplots()

ax.scatter(x = train.GrLivArea, y= train.SalePrice )

ax.set_xlabel('GrLivArea',fontsize=13)

ax.set_ylabel('SalePrice',fontsize =13)

plt.show()
## Plot fig sizing. 

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train.corr(), 

            cmap=sns.diverging_palette(20, 220, n=200), 

            mask = mask, 

            annot=True, 

            center = 0, 

           );

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
numerical_features = train.dtypes[train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_features))



categorical_features = train.dtypes[train.dtypes == "object"].index

print("Number of categorical features:", len(categorical_features))
# Check the percentage of missing values for each columns



total = train.isnull().sum().sort_values(ascending=False)

percent_1 = train.isnull().sum()/train.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['total', 'Missing_Ratio'])

missing_data.head(20)
total = test.isnull().sum().sort_values(ascending=False)

percent_1 = test.isnull().sum()/test.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['total', 'Missing_Ratio'])

missing_data.head(20)
train['LotFrontage'] = train['LotFrontage'].fillna(train.LotFrontage.mean())



test['LotFrontage'] = test['LotFrontage'].fillna(test.LotFrontage.mean())
list1 =["Alley",

                       "PoolQC",

                      "MiscFeature",

                       "Fence",

                       "GarageCond",

                       "GarageQual",

                       "GarageFinish",

                       "GarageType",

                       "FireplaceQu",

                       "BsmtExposure",

                       "BsmtCond",

                       "BsmtQual",

                       "BsmtFinType1",

                       "BsmtFinType2",

                       "MasVnrType"]

                       

for i in list1:

    

    train[i] = train[i].fillna("None")

    test[i] = test[i].fillna("None")
list2 = ["MasVnrArea",

                          "BsmtFinSF1",

                          "BsmtFinSF2",

                           "BsmtUnfSF",

                          "TotalBsmtSF",

                          "BsmtFullBath",

                          "BsmtHalfBath",

                          "GarageYrBlt",

                          "GarageCars",

                          "GarageArea"]



for i in list2:

    train[i] = train[i].fillna(0)

    test[i] = test[i].fillna(0)
train ["Utilities"]= train ["Utilities"].fillna("AllPub")

train ["Electrical"] = train ["Electrical"] .fillna("SBrkr")

train ["Functional"] = train ["Functional"] .fillna("Typ")
test ["Utilities"]=test ["Utilities"].fillna("AllPub")

test["Electrical"] = test["Electrical"] .fillna("SBrkr")

test ["Functional"] =test ["Functional"] .fillna("Typ")
train ["Exterior1st"]= train["Exterior1st"].fillna(train["Exterior1st"].mode()[0])

train ["Exterior2nd"]= train["Exterior2nd"].fillna(train["Exterior2nd"].mode()[0])

train["KitchenQual"] = train["KitchenQual"] .fillna(train["KitchenQual"].mode()[0])

train["SaleType"]  =   train["SaleType"] .fillna(train["SaleType"].mode()[0])
test ["Exterior1st"]= test["Exterior1st"].fillna(test["Exterior1st"].mode()[0])

test ["Exterior2nd"]= test["Exterior2nd"].fillna(test["Exterior2nd"].mode()[0])

test["KitchenQual"] = test["KitchenQual"] .fillna(test["KitchenQual"].mode()[0])

test["SaleType"]  =   test["SaleType"] .fillna(test["SaleType"].mode()[0])

test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])
# Some of the non-numeric predictors are stored as numbers; we convert them into strings 





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
train.shape, test.shape
train.isnull().any().any()
test.isnull().any().any()
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




n_folds = 5

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
#Lasso



lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
#ElasticNet



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

#KernelRidge



KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#Gradient Boosting Regressor



GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
# XGB Regressor



model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, verbosity=0,

                             random_state =7, nthread = -1)
#light gbm 



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
#rmsle 

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