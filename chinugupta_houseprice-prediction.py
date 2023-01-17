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
import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.preprocessing as skpe

import sklearn.model_selection as ms

import sklearn.metrics as sklm

import sklearn.ensemble as ensemble

import xgboost as xgb

import lightgbm as lgb

import scipy.stats as stats

import sklearn.kernel_ridge as ridge

import numpy.random as nr

import sklearn.linear_model as lm
path = "../input/house-prices-advanced-regression-techniques/train.csv"

train = pd.read_csv(path)

train.head()
path1 = "../input/house-prices-advanced-regression-techniques/test.csv"

test = pd.read_csv(path1)

test.head()
train.info()

print("-------------------------------------")



test.info()
train.describe()
test.describe()
# Let's look at the target variable first

train['SalePrice'].describe()
# This looks slightly right-skewed

sns.distplot(train['SalePrice'])
# Skewness and Kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
sns.scatterplot(y='SalePrice',x='LotArea',data=train)
sns.scatterplot(y='SalePrice',x='TotalBsmtSF',data=train)
sns.scatterplot(y='SalePrice',x='GrLivArea',data=train)
sns.boxplot(y=train['SalePrice'],x=train['OverallQual'])
# HeatMap

fig, ax = plt.subplots(figsize=(20, 18))

sns.heatmap(train.corr(), vmax=.8)
k = 10 #number of variables for heatmap

cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
corr = train.corr()



# Sort in descending order

corr_top = corr['SalePrice'].sort_values(ascending=False)[:10]

top_features = corr_top.index[1:]

print(corr_top)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'LotArea','YearBuilt']

sns.pairplot(train[cols], size = 2.5)
Q1 = []

Q3 = []

Lower_Bound = []

Upper_Bound = []

Outliers = []



for i in top_features:

    

    # 25th and 75th percentiles

    q1, q3 = np.percentile(train[i], 25), np.percentile(train[i], 75)

    

    # Interquartile range

    iqr = q3 - q1

    

    # Outlier cutoff

    cut_off = 1.5*iqr

    

    # Lower and upper bounds

    lower_bound = q1 - cut_off

    upper_bound = q3 + cut_off

    

    # Save outlier indexes

    outlier = [x for x in train.index if train.loc[x,i] < lower_bound or train.loc[x,i] > upper_bound]

    

    # Append values for dataframe

    Q1.append(q1)

    Q3.append(q3)

    Lower_Bound.append(lower_bound)

    Upper_Bound.append(upper_bound)

    Outliers.append(len(outlier))

    

    try:

        train.drop(outlier, inplace=True, axis=0)

        

    except:

        continue

        

df_out = pd.DataFrame({'column':top_features,'Q1':Q1,'Q3':Q3,'Lower_Bound':Lower_Bound,'Upper_Bound':Upper_Bound,'No. of Outliers':Outliers})

df_out.sort_values(by='No. of Outliers', ascending=False)
train.shape
# Saving train rows

ntrain = train.shape[0]



# Save log transformation of target variable to deal with the skewness

target = np.log(train['SalePrice'])



# Drop Id and SalePrice from train dataframe

train.drop(['Id', 'SalePrice'], inplace=True, axis=1)



# Store test Id

test_Id = test['Id']



# Drop test Id

test.drop(['Id'], inplace=True, axis=1)



# Concatenate train and test dataframes

train = pd.concat([train, test])
train.isnull().sum().sort_values(ascending=False).head(40)
# Ordinal Features



# NA means no pool

train['PoolQC'].replace(['Ex', 'Gd', 'Fa', np.nan],[3,2,1,0], inplace=True)



# NA means no fence

train['Fence'].replace(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', np.nan],[4,3,2,1,0], inplace=True)



# NA means no fireplace

train['FireplaceQu'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan],[5,4,3,2,1,0], inplace=True)



# Garage Features

train['GarageCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan],[5,4,3,2,1,0], inplace=True)



train['GarageQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan],[5,4,3,2,1,0], inplace=True)



train['GarageFinish'].replace(['RFn', 'Fin', 'Unf', np.nan],[3,2,1,0], inplace=True)



# Bsmt Features

for i in ['BsmtCond', 'BsmtQual']:

    train[i].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan],[5,4,3,2,1,0], inplace=True)



train['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No', np.nan],[4,3,2,1,0], inplace=True)



for i in ['BsmtFinType1', 'BsmtFinType2']:

    train[i].replace(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', np.nan],[6,5,4,3,2,1,0], inplace=True)





# Nominal Features



# NA means no alley

train['Alley'].fillna('None', inplace=True)



# NA means no miscellaneous features

train['MiscFeature'].fillna('None', inplace=True)



# NA means no garage type

train['GarageType'].fillna('None', inplace=True)



# NA means no masonry work

train['MasVnrType'].fillna('None', inplace=True)



# If no work, then no area

train['MasVnrArea'].fillna(0, inplace=True)





# Numerical Features



# Replace null lotfrontage with average of the neighbourhood

train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Filling 0 with null values in BsmtFeatures

for i in ['BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']:

    train[i].fillna(0, inplace=True)

    

# Replace with most common values

for i in ['MSZoning', 'Utilities', 'KitchenQual']:

    train[i].fillna(train[i].mode()[0], inplace=True)

    

train['Functional'].fillna('Typ', inplace=True)



train['SaleType'].fillna('Oth' ,inplace=True)



# Replace with most common value

train['Electrical'].fillna(train['Electrical'].mode()[0] ,inplace=True)



train['GarageCars'].fillna(train['GarageCars'].mode()[0] ,inplace=True)



train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode()[0] ,inplace=True)



# Repace with 'Other' value

for i in ['Exterior1st', 'Exterior2nd']:

    train[i].fillna('Other', inplace=True)

    

train['KitchenQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'],[4,3,2,1,0], inplace=True)



train['GarageArea'].fillna(train['GarageArea'].astype('float').mean(axis=0), inplace=True)

# Total surface area of house

train['TotalSF'] = train.apply(lambda x: x['1stFlrSF'] + x['2ndFlrSF'] + x['TotalBsmtSF'], axis=1)



# Total bathrooms in the house

train['TotalBath'] = train.apply(lambda x: x['FullBath'] + 0.5*x['HalfBath'] + x['BsmtFullBath'] + 0.5*x['BsmtHalfBath'], axis=1)



# Total porch area of house

train['TotalPorch'] = train.apply(lambda x: x['OpenPorchSF'] + x['EnclosedPorch'] + x['3SsnPorch'] + x['ScreenPorch'], axis=1)
# Dummifying the dataset for modelling

train =pd.get_dummies(train, drop_first=True)

train.shape
# Train dataset

df = train.iloc[:ntrain,:]



# Test dataset

test = train.iloc[ntrain:,:]



# Seperating independent and dependent variables

X = df

y = target
# train,test split to get training,validation and testing

X_train,X_test,y_train,y_test = ms.train_test_split(X,y,random_state=2,test_size=0.2)
lr = lm.LinearRegression()

lr.fit(X_train,y_train)



rmse = np.sqrt(sklm.mean_squared_error(y_test,lr.predict(X_test)))

print(rmse)
# Different alpha values

alphas = [0.01,0.1,0.3,1,3,5,10,20]



for a in alphas:

    kernel_ridge = ridge.KernelRidge(alpha=a)

    kernel_ridge.fit(X_train,y_train)

    

    rmse = np.sqrt(sklm.mean_squared_error(y_test,kernel_ridge.predict(X_test)))

    print('For alpha =',a,',','RMSE = ',rmse)
model = ridge.KernelRidge(alpha=0.1)

model.fit(X_train,y_train)
log_pred = model.predict(test)

actual_pred = np.exp(log_pred)
subm_dict = {'Id':test_Id, 'SalePrice':actual_pred}

submit = pd.DataFrame(subm_dict)

submit.to_csv('submission.csv', index=False)