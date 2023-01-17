# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

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



from sklearn import metrics

from sklearn.model_selection import train_test_split



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

text = pd.read_fwf('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train_ID = train['Id']

test_ID = test['Id']



train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
train.head()
train.shape
test.head()
test.shape
submission.head()
train.drop('SalePrice', axis=1)

for i in train.drop('SalePrice', axis=1).columns:

    if train[i].dtype == 'int64':

        train[i] = train[i].astype('int32')

    elif train[i].dtype == 'float64':

        train[i] = train[i].astype('float32')
numerical_feats = train.dtypes[train.dtypes != 'object'].index

numerical_feats = numerical_feats.drop('SalePrice')

print("Numerical features: ", len(numerical_feats))



categorical_feats = train.dtypes[train.dtypes == 'object'].index

print("Categorical features: ", len(categorical_feats))
train_num = train[numerical_feats]

train_cat = train[categorical_feats]
train['SalePrice'].describe()
sns.distplot(train.SalePrice)
print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
numerical_feats = train.dtypes[train.dtypes != 'object'].index

print("Numerical features: ", len(numerical_feats))



categorical_feats = train.dtypes[train.dtypes == 'object'].index

print("Categorical features: ", len(categorical_feats))
num_cont = []

num_disc = []

for col in numerical_feats:

    if train[col].nunique() > 25: 

        num_cont.append(col)

    else:

        num_disc.append(col)
train.hist(num_cont,bins=50, figsize=(25,20))

plt.tight_layout(pad=0.4)

plt.show()
plt.figure(figsize = (16,50))

for idx,col in enumerate(num_disc):

    plt.subplot(9,2,idx+1)

    ax=sns.countplot(train[col])
plt.figure(figsize = (20,100))

for idx,col in enumerate(categorical_feats):

    plt.subplot(22,2,idx+1)

    ax=sns.countplot(train[col])
for col in categorical_feats:

    print(train[col].value_counts())

    print("\n")
vars = train_num.columns

figures_per_time = 4

count = 0 

y = train.SalePrice

for var in vars:

    x = train[var]

    plt.figure(count//figures_per_time,figsize=(25,5))

    plt.subplot(1,figures_per_time,np.mod(count,4)+1)

    plt.scatter(x, y);

    plt.title('f model: T= {}'.format(var))

    count+=1

    
corrmat = train.corr()

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(corrmat, vmax=.9, square=True);
corr_num = 11

cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index

corr_mat_sales = np.corrcoef(train[cols_corr].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(12, 9))

hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)

plt.show()
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = plt.scatter(x=train['OverallQual'], y="SalePrice", data=data)
data = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)

f, ax = plt.subplots(figsize=(30, 8))

fig = sns.boxplot(x=train['YearBuilt'], y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=45);
data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = plt.scatter(x=train['TotalBsmtSF'], y="SalePrice", data=data)
data = pd.concat([train['SalePrice'], train['LotArea']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = plt.scatter(x=train['LotArea'], y="SalePrice", data=data)
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = plt.scatter(x=train['GrLivArea'], y="SalePrice", data=data)
train.loc[(train['OverallQual']<5) & (train['SalePrice']>200000)]
train.loc[(train['OverallQual']<9) & (train['SalePrice']>500000)]
train.loc[(train['GrLivArea']>4500) & (train['SalePrice']<300000)]
train.loc[(train['TotalBsmtSF']>6000) & (train['SalePrice']>150000)]
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)

train.drop(train[(train['OverallQual']<9) & (train['SalePrice']>500000)].index, inplace=True)

train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)

train.reset_index(drop=True, inplace=True)
ntrain = train.shape[0]

ntest = test.shape[0]

y = train.SalePrice.values

all_data = pd.concat((train,test)).reset_index(drop=True)

all_data.drop('SalePrice', axis = 1, inplace=True)

all_data.shape
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
print("Null Values:\n",all_data.isnull().sum())
all_data_null = all_data.isnull().sum()

all_data_null = all_data_null[all_data_null > 0]
all_data_null.sort_values(inplace=True)

all_data_null
def handle_missing(features):

    # the data description states that NA refers to typical ('Typ') values

    features['Functional'] = features['Functional'].fillna('Typ')

    # Replace the missing values in each of the columns below with their mode

    features['Electrical'] = features['Electrical'].fillna("SBrkr")

    features['KitchenQual'] = features['KitchenQual'].fillna("TA")

    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    

    # the data description stats that NA refers to "No Pool"

    features["PoolQC"] = features["PoolQC"].fillna("None")

    # Replacing the missing values with 0, since no garage = no cars in garage

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

        features[col] = features[col].fillna(0)

    # Replacing the missing values with None

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

        features[col] = features[col].fillna('None')

    # NaN values for these categorical basement features, means there's no basement

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

        features[col] = features[col].fillna('None')

        

    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood

    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



    # We have no particular intuition around how to fill in the rest of the categorical features

    # So we replace their missing values with None

    objects = []

    for i in features.columns:

        if features[i].dtype == object:

            objects.append(i)

    features.update(features[objects].fillna('None'))

        

    # And we do the same thing for numerical features, but this time with 0s

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numeric = []

    for i in features.columns:

        if features[i].dtype in numeric_dtypes:

            numeric.append(i)

    features.update(features[numeric].fillna(0))    

    return features
all_data = handle_missing(all_data)
all_data.isnull().sum().sum()
years = []

for col in all_data.columns:

    if "Y" in col or "Year" in col:

        years.append(col)



years = set(years)

years
plt.figure(figsize = (15,12))

for idx,col in enumerate(years):

    plt.subplot(2,2,idx+1)

    plt.plot(train.groupby(col)["SalePrice"].median())

    plt.xlabel(col)

    plt.ylabel("SalePrice")
numerical_feats_all = all_data.dtypes[all_data.dtypes != 'object'].index

print("Numerical features: ", len(numerical_feats_all))



categorical_feats_all = all_data.dtypes[all_data.dtypes == 'object'].index

print("Categorical features: ", len(categorical_feats_all))
all_num = all_data[numerical_feats_all]

all_cat = all_data[categorical_feats_all]
from scipy.stats import skew 

skewness = all_num.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
skew_features = all_num.apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features.head(10)
for col in skew_features.keys().to_list():

    all_data[col] = np.log1p(all_data[col])
all_data['YearsSinceRemodel'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd'].astype(int)

all_data['Total_Home_Quality'] = all_data['OverallQual'] + all_data['OverallCond']

all_data = all_data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']



all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'])

all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])
def addCrossSquared(temp, plist):

    m = temp.shape[1]

    for i in range(len(plist)-1):

        for j in range(i+1,len(plist)):

            temp = temp.assign(newcol=pd.Series(temp[plist[i]]*temp[plist[j]]).values)   

            temp.columns.values[m] = plist[i] + '*' + plist[j]

            m += 1

    return temp
poly_features_list = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF','1stFlrSF']

all_data = addCrossSquared(all_data, poly_features_list)
all_data = pd.get_dummies(all_data).reset_index(drop=True)

all_data.shape
all_data.head()
all_data['MSZoning_C (all)'].value_counts()

all_data.drop('MSZoning_C (all)', axis=1, inplace=True)
train_data = all_data[:ntrain]

test_data = all_data[ntrain:]
X = train_data.values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=2019)

y_train = np.log(y_train)
from xgboost.sklearn import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from ml_metrics import rmsle
Models = pd.DataFrame({'Model': [],'RMSLE': []})
best_xgb = XGBRegressor(learning_rate=0.001,n_estimators=6000,

                                max_depth=7, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)

best_xgb.fit(X_train,y_train)

y_pred = best_xgb.predict(X_test)

y_pred = np.exp(y_pred)



res = pd.DataFrame({"Model":['XGBoost'],

                    "RMSLE": [rmsle(y_test, y_pred)]})

Models = Models.append(res)
best_dtr = DecisionTreeRegressor(max_depth=4)

best_dtr.fit(X_train,y_train)

y_pred = best_dtr.predict(X_test)

y_pred = np.exp(y_pred)



print(rmsle(y_test, y_pred))



res = pd.DataFrame({"Model":['Decision Tree'],

                    "RMSLE": [rmsle(y_test, y_pred)]})

Models = Models.append(res)
best_rf = RandomForestRegressor(n_estimators=1500,

                                max_depth=6)

best_rf.fit(X_train,y_train)

y_pred = best_rf.predict(X_test)

y_pred = np.exp(y_pred)

print(rmsle(y_test, y_pred))



res = pd.DataFrame({"Model":['Random Forest'],

                    "RMSLE": [rmsle(y_test, y_pred)]})

Models = Models.append(res)
Models