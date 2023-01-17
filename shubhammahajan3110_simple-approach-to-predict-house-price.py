import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame

import matplotlib.pyplot as plt  # for plots

%matplotlib inline

import seaborn as sns # For advance plots and graphs





# Ignore Warnings

import warnings

warnings.filterwarnings("ignore")



# to view all columns

from IPython.display import display

pd.options.display.max_columns = None



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# combine train and test data set

data = pd.concat([train, test],ignore_index=True, sort=False)
print('the shape of  train dataset', train.shape)

print('the shape of  test dataset', test.shape)

print('the shape of data', data.shape)
train.head()
train.tail()
data.info()
data.describe().T
train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
sns.distplot(train['SalePrice'])

plt.show()
print("Skewness in SalePrice :", train['SalePrice'].skew())

print("Kurtosis in SalePrice :", train['SalePrice'].kurt())
sns.boxplot(train['SalePrice'])

plt.show()
train['SalePrice'] = np.log(train["SalePrice"]+1)

#check distribution after log transform.

sns.distplot(train['SalePrice'])

plt.show()
#check correlation in features

cor = train.corr()

plt.figure(figsize=(15,10))

sns.heatmap(cor,cmap="Blues", vmax=0.9)

plt.show()
# checking high correlatinal features to 'SalePrice'

cor[abs(cor['SalePrice'].values) >= 0.5]['SalePrice'].sort_values(ascending=False)[1:]
# OverllQual is catagorical feature so draw boxplot

fig = plt.figure(constrained_layout=True, figsize=(8,5))

sns.boxplot(train['OverallQual'], train['SalePrice'])

plt.show()
fig = plt.figure(constrained_layout=True, figsize=(8,5))

sns.scatterplot(train['GrLivArea'], train['SalePrice'])

plt.show()
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<12.5)].index)



train.reset_index(drop = True, inplace = True)
# sctter plot after removing outliers.

fig = plt.figure(constrained_layout=True, figsize=(8,5))

sns.scatterplot(train['GrLivArea'], train['SalePrice'])

plt.show()
fig = plt.figure(constrained_layout=True, figsize=(8,5))

sns.boxplot(train['GarageCars'], train['SalePrice'])

plt.show()
sns.scatterplot(train['GarageArea'], train['SalePrice'])

plt.show()
sns.scatterplot(train['TotalBsmtSF'], train['SalePrice'])

plt.show()
fig = plt.figure(constrained_layout=True, figsize=(8,5))

sns.boxplot(train['FullBath'], train['SalePrice'])

plt.show()
## Plot fig sizing. 

import matplotlib.style as style

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, );

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
# again concat train and test cause we have done changes in train dataset. 

data = pd.concat((train, test)).reset_index(drop=True)
columns = data.columns



for col in columns:

    print(data[col].value_counts())

    print('\n')
missing_tot = data.isnull().sum().sort_values(ascending = False)

missing_per = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([missing_tot, missing_per] , axis=1, keys=['Total', 'Percent'])
missing_data.head(40)
data_cat = data.select_dtypes('object')

data_int =  data.select_dtypes(['int64','float64'])
data_cat.columns
data_cat.isnull().sum().sort_values(ascending =False)
d = data_cat.isnull().sum().sort_values(ascending =False)

d.index
# fill null values by none cause this null represpents they are not avaliable on site so fill by 'None'

columns1 = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',

       'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure',

       'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']



# fill by mode

columns2 = ['MSZoning','Functional', 'Utilities', 'Electrical', 'KitchenQual', 'SaleType',

       'Exterior2nd', 'Exterior1st']
for col in columns1:

    data_cat[col].fillna('None',inplace =True)
for col  in columns2:

    data_cat[col].fillna(data[col].mode()[0],inplace = True)
d2 = data_int.isnull().sum().sort_values(ascending=False)

d2.index
columns3 = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtFullBath',

       'BsmtHalfBath', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageArea', 'GarageCars',

       'BsmtFinSF2', 'BsmtFinSF1']
for col in columns3:

    data_int[col].fillna(0,inplace=True)
df = pd.concat([data_cat,data_int],axis=1)
# Since these column are actually a category , using a numerical number will lead the model to assume

# that it is numerical , so we convert to string .

df['MSSubClass'] = df['MSSubClass'].apply(str)

df['YrSold'] = df['YrSold'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)

df.drop('SalePrice', axis=1, inplace = True)
missing_tot = df.isnull().sum().sort_values(ascending = False)

missing_per = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)

missing_data = pd.concat([missing_tot, missing_per] , axis=1, keys=['Total', 'Percent'])

missing_data.head()
numeric_feats = df.dtypes[df.dtypes != "object"].index



skewed_feats = df[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)

skewed_feats
from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



high_skew = skewed_feats[skewed_feats > 0.5]

skew_index = high_skew.index



# Normalise skewed features

for i in skew_index:

    df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

    
final_features = pd.get_dummies(df).reset_index(drop=True)

print('Features size:', df.shape)

final_features.head()
final_features.shape
nrow_train = train.shape[0]



X_train = final_features[:nrow_train]

X_test = final_features[nrow_train:]

Y = train['SalePrice']
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score



e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



# Kernel Ridge Regression : made robust to outliers

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))



# LASSO Regression : made robust to outliers

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 

                    alphas=alphas2,random_state=42, cv=kfolds))



# Elastic Net Regression : made robust to outliers

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 

                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))



#Fit the training data X_train,Y 

print('Elasticnet')

elastic_model = elasticnet.fit(X_train, Y)

print('Lasso')

lasso_model = lasso.fit(X_train, Y)

print('Ridge')

ridge_model = ridge.fit(X_train, Y)
# model blending function using fitted models to make predictions

def blend_models(X):

    return ((elastic_model.predict(X)) + (lasso_model.predict(X)) + (ridge_model.predict(X)))/3

submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.expm1(blend_models(X_test))
# Fix outleir predictions

q1 = submission['SalePrice'].quantile(0.0045)

q2 = submission['SalePrice'].quantile(0.99)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission_regression1.csv", index=False)
submission