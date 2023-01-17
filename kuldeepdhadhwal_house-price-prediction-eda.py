import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import missingno as msno

import matplotlib.pyplot as plt

from scipy.stats import skew

%matplotlib inline



sns.set_style("whitegrid")

import os
path = '/kaggle/input/home-data-for-ml-course/'
train_df = pd.read_csv(path+'train.csv')

test_df = pd.read_csv(path+'/test.csv')

sub_df = pd.read_csv(path+'/sample_submission.csv')
train_df.head()
test_df.head()
train_df.columns
test_df.columns
train_df.SalePrice.describe()
plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

sns.distplot(train_df.SalePrice, bins=50)

plt.title('Original')



plt.subplot(1,2,2)

sns.distplot(np.log1p(train_df.SalePrice), bins=50)

plt.title('Log transformed')

train_df.SalePrice.skew()
train_df.SalePrice.kurt()
train_df['GrLivArea']
var = 'GrLivArea'

data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)

data.head()
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
corr_matrix = train_df.corr()
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_matrix, vmax=.8, square=True)

sns.heatmap
k = 10 #number of variables for heatmap

cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_df[cols], size = 2.5)

plt.show()
total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum() / train_df.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
msno.matrix(train_df.sample(500))
msno.bar(train_df)
msno.heatmap(train_df)
target = train_df['SalePrice']

target_log = np.log1p(train_df['SalePrice'])
# drop target variable from train dataset

train = train_df.drop(["SalePrice"], axis=1)

data = pd.concat([train, test_df], ignore_index=True)
data.head()
# save all categorical columns in list

categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']



# dataframe with categorical features

data_cat = data[categorical_columns]

# dataframe with numerical features

data_num = data.drop(categorical_columns, axis=1)
data_num.head(1)
data_num.describe()
data_cat.head(1)
data_num.head()
data_num_skew = data_num.apply(lambda x: skew(x.dropna()))

data_num_skew = data_num_skew[data_num_skew > .75]



# apply log + 1 transformation for all numeric features with skewnes over .75

data_num[data_num_skew.index] = np.log1p(data_num[data_num_skew.index])
data_num_skew
data_num.drop
data_len = data_num.shape[0]



# check what is percentage of missing values in categorical dataframe

for col in data_num.columns.values:

    missing_values = data_num[col].isnull().sum()

    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 

    

    # drop column if there is more than 50 missing values

    if missing_values > 50:

        #print("droping column: {}".format(col))

        data_num = data_num.drop(col, axis = 1)

    # if there is less than 50 missing values than fill in with median valu of column

    else:

        #print("filling missing values with median in column: {}".format(col))

        data_num = data_num.fillna(data_num[col].median())
data_len = data_cat.shape[0]



# check what is percentage of missing values in categorical dataframe

for col in data_cat.columns.values:

    missing_values = data_cat[col].isnull().sum()

    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 

    

    # drop column if there is more than 50 missing values

    if missing_values > 50:

        print("{}".format(col))

        data_cat.drop(col, axis = 1)

    # if there is less than 50 missing values than fill in with median valu of column

    else:

        #print("filling missing values with XXX: {}".format(col))

        #data_cat = data_cat.fillna('XXX')

        pass
data_cat.describe()

columns = ['Alley',

'BsmtQual',

'BsmtCond',

'BsmtExposure',

'BsmtFinType1',

'BsmtFinType2',

'FireplaceQu',

'GarageType',

'GarageFinish',

'GarageQual',

'GarageCond',

'PoolQC',

'Fence',

'MiscFeature'

]

data_cat = data_cat.drop(columns, axis=1)
data_cat.head()
data_cat.dropna()
data_num.describe()
data_num.columns
data_cat = pd.get_dummies(data_cat)
frames = [data_cat, data_num]

total_df = pd.concat(frames,  axis=1)
total_df.head()
# target.head()

sub_df.head()
import xgboost as xgb

from sklearn.metrics import mean_squared_error
train_df = total_df[:1460]

test_df = total_df[1461:]
train_df.tail()
test_df.tail()
sub_df.head()
test_df.head()
target.tail()