# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn import ensemble

from scipy.stats import skew

import pandas_profiling

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.externals import joblib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

#train_data_overview.to_file(outputfile="House Price Pred data profiling.html")

                             
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True) 
import seaborn as sns

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
from scipy import stats

from scipy.stats import norm, skew #for some statistics



sns.distplot(train['SalePrice'])



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
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

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
#outliers

sns.lmplot(data = train, x = 'GrLivArea', y = 'SalePrice')
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice']<300000)].index)

sns.lmplot(data = train, x = 'GrLivArea', y = 'SalePrice')
# Concat all the data together

ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
# For columns if very few nulls, replace the nulls with the mode

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
all_data = all_data.drop(['Utilities'], axis=1)


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')



# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
# For these bathroom features, they are numerical, so NaNs mean there isnt a Bathroom for the basment so '0'

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
# For these Basement features, NaN means there is no basement so fill with 'None'

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
cat_feats = all_data.select_dtypes(include = 'object')

num_feats = all_data.select_dtypes(exclude = 'object')
from sklearn.preprocessing import LabelEncoder

# process columns, apply LabelEncoder to categorical features

for c in cat_feats:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))
# shape        

print('Shape all_data: {}'.format(all_data.shape))
all_data = pd.get_dummies(all_data)

print(all_data.shape)
def crossval(X, y, reg):

    result = []

    for i in range(10):

        result.append(cross_val_score(reg, X, y).mean())

    return np.mean(result)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import numpy as np

import warnings

warnings.filterwarnings("ignore")
train = all_data[:ntrain]

test = all_data[ntrain:]
X = train

y = y_train

reg = RandomForestRegressor()

crossval(X, y,reg)
result = []

for i in np.arange(1,10):

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33, random_state = i)

    xgb = XGBRegressor(learning_rate = 0.08)

    xgb.fit(X_train, Y_train, verbose=False)

    score = xgb.score(X_test,Y_test)

    result.append(score)

np.mean(result)
results = xgb.predict(test)



results = np.expm1(results)



submit = pd.Series(results, index= test_ID, name='SalePrice')

submit.to_csv("model.csv", header=True)