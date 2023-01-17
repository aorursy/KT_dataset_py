# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
print(train.shape, test.shape, submission.shape)
#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
train.head(0)
print("Before drop ID", train.shape, test.shape)

train_ID = train['Id']

test_ID = test['Id']



train = train.drop(['Id'], axis = 1)

test = test.drop(['Id'], axis=1)

print("After drop ID", train.shape, test.shape)



fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()

#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
all_data = pd.concat((train, test)).reset_index(drop=True)
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

y_train = train['SalePrice'].values



all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)



print(all_data.shape)

all_data_na = (all_data.isnull().sum() / len(all_data))*100

all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index)
all_data_na
all_data_na.index
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#Correlation map to see how features are correlated with SalePrice

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
corrmat = train.corr()

corrmat


fig, ax = plt.subplots()

ax.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)

plt.show()
all_data['PoolQC'].unique()
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['PoolQC'].unique()
all_data['MiscFeature'].unique()
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data['Neighborhood'].unique()
all_data['Alley'].unique()
all_data.groupby("Neighborhood")["LotFrontage"]
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 

            'GarageArea', 

            'GarageCars',

            'BsmtFinSF1', 

            'BsmtFinSF2', 

            'BsmtUnfSF',

            'TotalBsmtSF', 

            'BsmtFullBath', 

            'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
all_data['Electrical'].mode()
all_data['Electrical'].fillna(all_data['Electrical'].mode())

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data = all_data.drop(['Utilities'], axis=1)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data_na = all_data.isnull().sum() / len(all_data)

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data
all_data_na
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
list(all_data['BsmtQual'].values)
all_data['BsmtQual']
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
pd.set_option('display.max_columns', 10000)

all_data.head()
all_data.head()
str_line = []

column_name = []

unique_list = []



total = all_data



for i in total.columns:

    total_list1 = list(total[i].unique())

    total_list2 = list((pd.DataFrame(total_list1).dropna())[0])

    str_data = int(str(type(total_list2[0])) == "<class 'str'>")

    str_line.append(str_data)

    column_name.append(i)



str_line = pd.DataFrame(str_line)

column_name = pd.DataFrame(column_name)

all_column = pd.concat([column_name, str_line],1)

all_column.columns = ['column_name', 'strn']



str_column = pd.DataFrame(all_column[all_column['strn'] == 1])

str_column = list(str_column['column_name'])



unique_count = []

unique_column = []



for i in str_column:

    total_unique = list(total[i].unique())

    total_unique = len(total_unique)

    unique_count.append(total_unique)

    unique_column.append(i)



unique_count = pd.DataFrame(unique_count)

unique_column = pd.DataFrame(unique_column)

unique_total = pd.concat([unique_column, unique_count],1)



unique_total.columns = ['unique_column', 'unique_count']



unique_total = unique_total.sort_values(["unique_count"], ascending=[False])

# unique_total

# 상위 4개 항목들은 따로 관리해야 할듯, 나머지는 OHE진행



print(unique_total.unique_column.values)

unique_column = unique_total.unique_column.values
from sklearn.preprocessing import LabelEncoder

cols = unique_column

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
import lightgbm as lgb
all_data.head()
all_data[:10]
test.head()
train = all_data[:ntrain]

X = train

y = y_train



X_test = all_data[ntrain:]
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split (X,y, random_state=0)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X_train, y_train)
pred_train = model_lgb.predict(X_train)

pred_valid = model_lgb.predict(X_valid)



pred_train_expm = np.expm1(pred_train)

pred_valid_expm = np.expm1(pred_valid)

y_train_expm = np.expm1(y_train)

y_valid_expm = np.expm1(y_valid)
def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())



print(rmse(pred_train, y_train))

print(rmse(pred_valid, y_valid))
pred_test = model_lgb.predict(X_test)



pred_test_expm = np.expm1(pred_test)
submission.head()
submission = submission.drop("SalePrice",1)

pred_test_expm = pd.DataFrame(pred_test_expm)



submission_final = pd.concat([submission,pred_test_expm],axis=1)



submission_final.columns = ['ID','SalePrice']

submission_final.to_csv("submission_fianl.csv", index=False)

submission_final.tail()