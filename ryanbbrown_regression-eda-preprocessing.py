# import libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from scipy import stats

from scipy.stats import norm, skew



pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x)) # limit floats
# import data and save + drop the 'Id' column

og_df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

og_df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



train_ID = df_train['Id']

test_ID = df_test['Id']



df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)



print(df_train.shape)

print(df_test.shape)
df_train.columns
# descriptive statistics summary

df_train['SalePrice'].describe()
# histogram

sns.distplot(df_train['SalePrice'], fit=norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the normal distribution with those parameters to compare

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()



    # positive skew

    # peakedness
# skewness and kurtosis

print('Skewness: {:.4f}'.format(df_train['SalePrice'].skew()))

print('Kurtosis: {:.4f}'.format(df_train['SalePrice'].kurt()))
# log1p applies log(1+x) to all column elements

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])





# repeat code to look at new distribution

sns.distplot(df_train['SalePrice'], fit=norm)



(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
# general view of SalePrice values along normal dist

saleprice_scaled = StandardScaler().fit_transform(og_df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
plt.scatter(x = og_df_train['GrLivArea'], y = og_df_train['SalePrice'])

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')

plt.show()



# this is just a feature that we know shows clear outliers, there could be multiple
# deleting outliers

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<np.log1p(300000))].index)



# check the (log transformed) graphic

plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')

plt.show()
# concatenate the train and test data

y_train = df_train.SalePrice.values

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
# categorical, hence "None"

for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',

           'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',

            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):

    all_data[col] = all_data[col].fillna('None')



    

# numerical, hence 0

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',

            'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):

    all_data[col] = all_data[col].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
# Group by neighborhood and fill in missing value w median LotFrontage of all neighborhoods

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



# drop feature because it's the same for everyone

all_data = all_data.drop(['Utilities'], axis=1)



# data description says that NA means typical

all_data["Functional"] = all_data["Functional"].fillna("Typ")
#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
# we just change the type to str so dummy variables inclue them





#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
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
# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



# many more features can be added through similar methods
numeric_features = all_data.dtypes[all_data.dtypes != 'object'].index



# check skew of numerical features

skewed_features = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew':skewed_features})

skewness.head(10)
# box cox or log transformation



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15



for feat in skewed_features:

    all_data[feat] = boxcox1p(all_data[feat], lam)

    

    # could be np.log1p instead of boxcox1p
# can wait until after heatmap and such, redo of next step required

all_data = pd.get_dummies(all_data)

print(all_data.shape)
# re-separate for train and test

df_train = all_data[:1458]

df_test = all_data[1458:]

df_train['SalePrice'] = y_train



print(df_train.shape)

print(df_test.shape)
df_train.to_csv('df_train.csv', index=False)

df_test.to_csv('df_test.csv', index=False)
# top 40 correlation matrix

f, ax = plt.subplots(figsize=(12, 9))

k = 40

cols = df_train.corr().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)



sns.heatmap(cm, vmax=.8, square=True,yticklabels=cols.values, xticklabels=cols.values)



# # full heatmap

# f, ax = plt.subplots(figsize=(12, 9))

# sns.heatmap(df_train.corr(), vmax=.8, square=True)
# SalePrice zoomed heatmap

k = 10 #number of variables for heatmap

cols = df_train.corr().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
cols = ['SalePrice', 'OverallQual', 'TotalSF', 'GarageCars', 'FullBath', 'TotRmsAbvGrd']

sns.pairplot(df_train[cols], size = 2.5)