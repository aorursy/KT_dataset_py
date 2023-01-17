!pip install dabl
#Environment Setup:



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use('ggplot')



import dabl

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000

pd.options.display.max_columns = 8000
#import libraries:



#Data Pre-Processing



from sklearn.model_selection import train_test_split



from sklearn.model_selection import cross_val_score, GridSearchCV, KFold



from sklearn.preprocessing import RobustScaler, StandardScaler



from sklearn.pipeline import make_pipeline



from collections import Counter



from sklearn.preprocessing import scale



from sklearn.preprocessing import LabelEncoder



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Read, Data Visualization,EDA Analysis,Data Pre-Processing,Feature Engineering,Data Splitting
#Data Read

file_path = '../input/house-prices-advanced-regression-techniques'

train=pd.read_csv(f'{file_path}/train.csv')

test= pd.read_csv(f'{file_path}/test.csv')
train.shape, test.shape
train.head()
test.head()
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
plt.figure(figsize=(15,12))

sns.boxplot(train.YearBuilt, train.SalePrice)
plt.figure(figsize=(12,6))

plt.scatter(x=train.GrLivArea, y=train.SalePrice)

plt.xlabel("GrLivArea", fontsize=13)

plt.ylabel("SalePrice", fontsize=13)

plt.ylim(0,800000)
train.isna().sum()
test.isna().sum()
train_data=train.copy()
# Remove outliers from OverallQual, GrLivArea and SalesPrice

train_data.drop(train_data[(train_data['OverallQual']<5) & (train_data['SalePrice']>200000)].index, inplace=True)



train_data.drop(train_data[(train_data['GrLivArea']>4500) & (train_data['SalePrice']<300000)].index, inplace=True)



train_data.reset_index(drop=True, inplace=True)
train=train_data.copy()
#data transformation

train['SalePrice']=np.log(train['SalePrice'])
#lest we how its look after removing outliers

sns.scatterplot(x='GrLivArea',y='SalePrice',data=train)
ntrain = train.shape[0]

ntest = test.shape[0]



y_train = train.SalePrice.values



df = pd.concat((train, test)).reset_index(drop=True)
df.head()
#copying sales priece

y=df['SalePrice'].copy()
df.drop(['SalePrice'], axis=1, inplace=True)

print("df size is : {}".format(df.shape))
df1 = (df.isnull().sum() / len(df)) * 100

df1 = df1.drop(df1[df1 == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :df1})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=df1.index, y=df1)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
data=df.copy()
test_data=test.copy()
# These columns have a lot of Null values, so we drop them

data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

data.head()
#EDA Analysis Part:1
plt.style.use("fivethirtyeight")

plt.figure(figsize=(16, 9))

sns.distplot(data['MSSubClass'])

plt.xlabel("Type of Dwelling")

plt.ylabel("Count")

plt.title("Dwelling Type Count")

plt.show()
plt.style.use("ggplot")

plt.figure(figsize=(16, 9))

sns.countplot(data['MSZoning'])

plt.xlabel("Type of Zoning of the property")

plt.ylabel("Count")

plt.title("Zone Type Count")

plt.show()
plt.style.use("classic")

plt.figure(figsize=(16, 9))

sns.distplot(data['LotFrontage'])

plt.xlabel("Lot Frontage (in ft)")

plt.ylabel("Count")

plt.title("Lot Frontage Distribution")

plt.show()
plt.style.use("classic")

plt.figure(figsize=(16, 9))

sns.distplot(y, color='red')

plt.xlabel("Price (in $)")

plt.ylabel("Count")

plt.title("Sales Price Distribution")

plt.show()
dabl.plot(train_data, target_col='SalePrice')
#Data Preprocessing:To get rid of the data skewness, we have to log shift the data. We can apply log(1+x) to out data to shift it at center.
train_labels = y.apply(lambda x: np.log(1+x))
#Let's replot Sales price, to see if the skewness is gone



plt.style.use("classic")

plt.figure(figsize=(16, 9))

sns.distplot(train_labels, color='red')

plt.xlabel("Price (in $)")

plt.ylabel("Count")

plt.title("Sales Price Distribution")

plt.show()
#Dealing with NuLL Values
data['MSSubClass'] = data['MSSubClass'].apply(str)

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)



# the data description states that NA refers to typical ('Typ') values

data['Functional'] = data['Functional'].fillna('Typ')

# Replace the missing values in each of the columns below with their mode

data['Electrical'] = data['Electrical'].fillna("SBrkr")

data['KitchenQual'] = data['KitchenQual'].fillna("TA")

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



# Replacing the missing values with 0, since no garage = no cars in garage

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    data[col] = data[col].fillna(0)

# Replacing the missing values with None

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    data[col] = data[col].fillna('None')

# NaN values for these categorical basement features, means there's no basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('None')



# Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood

data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# We have no particular intuition around how to fill in the rest of the categorical features

# So we replace their missing values with None

objects = []

for i in data.columns:

    if data[i].dtype == object:

        objects.append(i)

data.update(data[objects].fillna('None'))



# And we do the same thing for numerical features, but this time with 0s

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in data.columns:

    if data[i].dtype in numeric_dtypes:

        numeric.append(i)

data.update(data[numeric].fillna(0))
data.isna().sum()
df1=data.copy()
#Check remaining missing values if any 

all_data_na = (df1.isnull().sum() / len(df1)) * 100

df2 = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :df2})

missing_data.head()
#Encode Categorical Features
data.head()
df=data.copy()
# Make a list of all categorical columns

cols = ['MSZoning', 'Street', 'LotShape',

            'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

            'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',

            'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

            'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

            'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

            'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(df[c].values)) 

    df[c] = lbl.transform(list(df[c].values))



# shape        

print('Shape all_data: {}'.format(df.shape))
# Adding total sqfootage feature 

df['TotalSF'] = df['TotalBsmtSF'] +df['1stFlrSF'] + df['2ndFlrSF']
df.head()
# Get the dummy variables from them

#data = pd.get_dummies(data, columns=cat_cols)
# Recheck the shape of the data

df.shape
from scipy import stats

from scipy.stats import norm, skew #for some statistics
# Remove any repeated columns

df = df.iloc[:, ~df.columns.duplicated()]
df.shape
y_train = train.SalePrice.values

y_train
from scipy import stats

from scipy.stats import norm, skew #for some statistics
numeric_feats = df.dtypes[df.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
""""Box Cox Transformation of (highly) skewed features



We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+x .



Note that setting  λ=0  is equivalent to log1p used above for the target variable.

"""
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    

    df[feat] = boxcox1p(df[feat], lam)
#Getting dummy categorical features
df = pd.get_dummies(df)

print(df.shape)
df.info()
df.describe()
import pandas_profiling
# preparing profile report



profile_report = pandas_profiling.ProfileReport(df,minimal=True)

profile_report
#Correlation map to see how features are correlated with SalePrice

corrmat = df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
df.apply(lambda x: sum(x.isnull()),axis=0)
import plotly.express as px
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
train = df[:ntrain]

test = df[ntrain:]
train
test
y_train 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#Ml models



from sklearn.linear_model import Ridge



from sklearn.linear_model import Lasso



from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor



from sklearn.svm import SVR, LinearSVR



from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge



from sklearn.kernel_ridge import KernelRidge



from xgboost import XGBRegressor



import xgboost as xgb



import lightgbm as lgb



#Metrices

from sklearn.metrics import mean_squared_error

#Validation function

n_folds = 10



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=2020).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=42))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=9, 

                             min_child_weight=1.7817, n_estimators=2000,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.03, n_estimators=1500,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
Rd=Ridge()
score = rmsle_cv(Rd)

print("Rd score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


Rf=  RandomForestRegressor()
score = rmsle_cv(Rf)

print("Rf score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
GBR= GradientBoostingRegressor()
score = rmsle_cv(GBR)

print("GBR score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
Ext= ExtraTreesRegressor()
score = rmsle_cv(Ext)

print("Ext score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
Svmr=LinearSVR()
score = rmsle_cv(Svmr)

print("Svmr score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


BR=BayesianRidge()
score = rmsle_cv(BR)

print("BR score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
score = rmsle_cv(lasso)

print("lasso score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=2020)

        

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
stacked_averaged_models = StackingAveragedModels(base_models = (model_lgb,model_xgb),

                                                 meta_model = model_xgb)
score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
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
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.10 +

               xgb_train_pred*0.10 + lgb_train_pred*0.80 ))
ensemble = stacked_pred*0.1 + xgb_pred*0.1 + lgb_pred*0.8
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = ensemble

sub.to_csv('submission.csv',index=False)