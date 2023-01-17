# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()
df_train.describe()
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()
df_test.describe()
#check for number of sample and features

print(df_train.shape)

print(df_test.shape)
#save 'Id' column for evaluation:

train_id = df_train['Id']

test_id = df_test['Id']
#drop 'Id' column since it is not required in prediction process:

df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)

#recheck no of sample and features:

print(df_train.shape)

print(df_test.shape)
#scatter plot



fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.xlabel("GrLivArea", fontsize=15)

plt.ylabel("SalePrice", fontsize=15)

plt.show()
#deleting outliers



df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
#recheck in scatter plot



fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.xlabel("GrLivArea", fontsize=15)

plt.ylabel("SalePrice", fontsize=15)

plt.show()
sns.distplot(df_train['SalePrice'], fit=norm);



# Get the fitted parameters used by the function

mu, sigma = norm.fit(df_train['SalePrice'])

print("value of mu:  %.2f" %mu)

print("value of sigma:  %.2f" %sigma)



#now, plot the distribution 

plt.legend(['Normal Dist: ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency', fontsize=13)

plt.xlabel('SalePrice Distribution', fontsize=13)



#get also the Q-Q plot

fig = plt.figure(figsize=(10, 5))

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



sns.distplot(df_train['SalePrice'], fit=norm);



# Get the fitted parameters used by the function

mu, sigma = norm.fit(df_train['SalePrice'])

print("value of mu:  %.2f" %mu)

print("value of sigma:  %.2f" %sigma)



#now, plot the distribution 

plt.legend(['Normal Dist: ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency', fontsize=13)

plt.xlabel('SalePrice Distribution', fontsize=13)



#get also the Q-Q plot

fig = plt.figure(figsize=(10, 5))

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
n_train = df_train.shape[0]

n_test = df_test.shape[0]

y_train = df_train.SalePrice.values

main_df = pd.concat((df_train, df_test)).reset_index(drop=True)

main_df.drop(['SalePrice'], axis=1, inplace=True)

print(main_df.shape)
main_df_na = (main_df.isnull().sum() / len(main_df)) * 100

main_df_na = main_df_na.drop(main_df_na[main_df_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio': main_df_na})

print(missing_data.head(20))
#plotting bar plot for the missing data:

f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation = '90')

sns.barplot(x = main_df_na.index, y = main_df_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percentage of missing values', fontsize=15)

plt.title('Percentage missing data by feature', fontsize=20)
#plot the hitmap for indicating correlation to see how features are corrleated to SalePrice

cor = df_train.corr()

plt.subplots(figsize=(10, 10))

sns.heatmap(cor, vmax=0.9, square=True)
main_df['PoolQC'] = main_df['PoolQC'].fillna('None')
main_df['MiscFeature'] = main_df['MiscFeature'].fillna('None')
main_df['Alley'] = main_df['Alley'].fillna('None')
main_df['Fence'] = main_df['Fence'].fillna('None')
main_df['FireplaceQu'] = main_df['FireplaceQu'].fillna('None')
main_df["LotFrontage"] = main_df.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    main_df[col] = main_df[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    main_df[col] = main_df[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    main_df[col] = main_df[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    main_df[col] = main_df[col].fillna('None')
main_df["MasVnrType"] = main_df["MasVnrType"].fillna("None")

main_df["MasVnrArea"] = main_df["MasVnrArea"].fillna(0)
main_df['MSZoning'] = main_df['MSZoning'].fillna(main_df['MSZoning'].mode()[0])
main_df = main_df.drop(['Utilities'], axis=1)
main_df["Functional"] = main_df["Functional"].fillna("Typ")
main_df['Electrical'] = main_df['Electrical'].fillna(main_df['Electrical'].mode()[0])
main_df['KitchenQual'] = main_df['KitchenQual'].fillna(main_df['KitchenQual'].mode()[0])
main_df['Exterior1st'] = main_df['Exterior1st'].fillna(main_df['Exterior1st'].mode()[0])

main_df['Exterior2nd'] = main_df['Exterior2nd'].fillna(main_df['Exterior2nd'].mode()[0])
main_df['SaleType'] = main_df['SaleType'].fillna(main_df['SaleType'].mode()[0])
main_df['MSSubClass'] = main_df['MSSubClass'].fillna("None")
#Check remaining missing values if any 

main_df_na = (main_df.isnull().sum() / len(main_df)) * 100

main_df_na = main_df_na.drop(main_df_na[main_df_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio': main_df_na})

missing_data.head()
#MSSubClass=The building class

main_df['MSSubClass'] = main_df['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

main_df['OverallCond'] = main_df['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

main_df['YrSold'] = main_df['YrSold'].astype(str)

main_df['MoSold'] = main_df['MoSold'].astype(str)
#importing labelencoder from sklearn package:



from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(main_df[c].values)) 

    main_df[c] = lbl.transform(list(main_df[c].values))



# shape        

print('Shape all_data: {}'.format(main_df.shape))
# Adding total sqfootage feature 

main_df['TotalSF'] = main_df['TotalBsmtSF'] + main_df['1stFlrSF'] + main_df['2ndFlrSF']
numeric_feats = main_df.dtypes[main_df.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = main_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
main_df = pd.get_dummies(main_df)

print(main_df.shape)
train = main_df[:n_train]

test = main_df[n_train:]


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
#Validation function

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

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)

print("\nLasso score=> Mean: {:.4f} Std Dev: {:.4f}\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)

print("\nElastic Net score=> Mean: {:.4f} Std Dev: {:.4f}\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("\nKernel Ridge score=> Mean: {:.4f} Std Dev: {:.4f}\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("\nGradient Boost score=> Mean: {:.4f} Std Dev: {:.4f}\n".format(score.mean(), score.std()))
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
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: Mean: {:.4f} Std Dev: {:.4f}\n".format(score.mean(), score.std()))
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

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: Mean: {:.4f} Std Dev: {:.4f}".format(score.mean(), score.std()))
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
#RMSE on the entire Train data when averaging



print('RMSLE score on train data:', rmsle(y_train,stacked_train_pred*0.70 +

               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
final = stacked_pred*0.7 + xgb_pred*0.15 + lgb_pred*0.15
output = pd.DataFrame()

output['Id'] = test_id

output['SalePrice'] = final

output.to_csv('submission.csv',index=False)