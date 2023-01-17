import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

%matplotlib inline
# Read files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#importing the liberaries

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#display the top 5 values

train.head()
#shape of train data

train.shape
#you can also check the data set information using the info() command.

train.info()
#Analysis for numerical variable



train['SalePrice'].describe()

sns.distplot(train['SalePrice']);

#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
from scipy import stats

from scipy.stats import norm, skew #for some statistics



# Plot histogram and probability

fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train['SalePrice'] , fit=norm);

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1,2,2)

res = stats.probplot(train['SalePrice'], plot=plt)

plt.suptitle('Before transformation')



# Apply transformation

train.SalePrice = np.log1p(train.SalePrice )

# New prediction

y_train = train.SalePrice.values

y_train_orig = train.SalePrice





# Plot histogram and probability after transformation

fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train['SalePrice'] , fit=norm);

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1,2,2)

res = stats.probplot(train['SalePrice'], plot=plt)

plt.suptitle('After transformation')
# y_train_orig = train.SalePrice

# train.drop("SalePrice", axis = 1, inplace = True)

data_features = pd.concat((train, test), sort=False).reset_index(drop=True)

print(data_features.shape)



# print(train.SalePrice)
#Let's check if the data set has any missing values. 

data_features.columns[train.isnull().any()]
#plot of missing value attributes

plt.figure(figsize=(12, 6))

sns.heatmap(train.isnull())

plt.show()
#missing value counts in each of these columns

Isnull = train.isnull().sum()/len(data_features)*100

Isnull = Isnull[Isnull>0]

Isnull.sort_values(inplace=True, ascending=False)

Isnull
#Convert into dataframe

Isnull = Isnull.to_frame()

Isnull.columns = ['count']

Isnull.index.names = ['Name']

Isnull['Name'] = Isnull.index

#plot Missing values

plt.figure(figsize=(13, 5))

sns.set(style='whitegrid')

sns.barplot(x='Name', y='count', data=Isnull)

plt.xticks(rotation = 90)

plt.show()
#missing data percent plot, basically percent plot is for categorical columns



total = data_features.isnull().sum().sort_values(ascending=False)

percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#Corralation between train And test attributes
#Separate variable into new dataframe from original dataframe which has only numerical values

#there is 38 numerical attribute from 81 attributes

train_corr = data_features.select_dtypes(include=[np.number])
train_corr.shape

#Delete Id because that is not need for corralation plot

#del train_corr['Id']
#Coralation plot

corr = train_corr.corr()

plt.subplots(figsize=(20,9))

sns.heatmap(corr, annot=True)
#0.5 is nothing but a threshold value.

#It is good to take it 0.5 because your feautres which are fitting under this threshold will give good accuracy.



top_feature = corr.index[abs(corr['SalePrice']>0.5)]

plt.subplots(figsize=(12, 8))

top_corr = data_features[top_feature].corr()

sns.heatmap(top_corr, annot=True)

plt.show()

col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

sns.set(style='ticks')

sns.pairplot(data_features[col], height=2, kind='reg')
#unique value of OverallQual

data_features.OverallQual.unique()
sns.barplot(data_features.OverallQual, data_features.SalePrice)
#boxplot

plt.figure(figsize=(18, 8))

sns.boxplot(x=data_features.OverallQual, y=data_features.SalePrice)
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

sns.set(style='ticks')

sns.pairplot(data_features[col], height=2, kind='reg')
print("Find most important features relative to target")

corr = data_features.corr()

corr.sort_values(['SalePrice'], ascending=False, inplace=True)

corr.SalePrice
# PoolQC has missing value ratio is 99%+. So, there is fill by None

data_features['PoolQC'] = data_features['PoolQC'].fillna('None')
#Arround 50% missing values attributes have been fill by None

data_features['MiscFeature'] = data_features['MiscFeature'].fillna('None')

data_features['Alley'] = data_features['Alley'].fillna('None')

data_features['Fence'] = data_features['Fence'].fillna('None')

data_features['FireplaceQu'] = data_features['FireplaceQu'].fillna('None')

data_features['SaleCondition'] = data_features['SaleCondition'].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

data_features['LotFrontage'] = data_features.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    data_features[col] = data_features[col].fillna('None')
#GarageYrBlt, GarageArea and GarageCars these are replacing with zero

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

    data_features[col] = data_features[col].fillna(int(0))
#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):

    data_features[col] = data_features[col].fillna('None')
#MasVnrArea : replace with zero

data_features['MasVnrArea'] = data_features['MasVnrArea'].fillna(int(0))
#MasVnrType : replace with None

data_features['MasVnrType'] = data_features['MasVnrType'].fillna('None')
#There is put mode value 

data_features['Electrical'] = data_features['Electrical'].fillna(data_features['Electrical']).mode()[0]
#There is no need of Utilities

data_features = data_features.drop(['Utilities'], axis=1)
#Checking there is any null value or not

plt.figure(figsize=(10, 5))

sns.heatmap(data_features.isnull())
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',

        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 

        'SaleType', 'SaleCondition', 'Electrical', 'Heating')

from sklearn.preprocessing import LabelEncoder

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(data_features[c].values)) 

    data_features[c] = lbl.transform(list(data_features[c].values))
train = data_features.iloc[:len(y_train), :]

test = data_features.iloc[len(y_train):, :]

print(['Train data shpe: ',train.shape,'Prediction on (Sales price) shape: ', y_train.shape,'Test shape: ', test.shape])
#Take targate variable into y

y = train['SalePrice']
#Delete the saleprice

del train['SalePrice']
#Take their values in X and y

X = train.values

y = y.values
# Split data into train and test formate

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
#Train the model

from sklearn import linear_model

model = linear_model.LinearRegression()
#Fit the model

model.fit(X_train, y_train)
#Prediction

print("Predict value " + str(model.predict([X_test[142]])))

print("Real value " + str(y_test[142]))
#Score/Accuracy

print("Accuracy --> ", model.score(X_test, y_test)*100)
#Train the model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=1000)
#Fit

model.fit(X_train, y_train)
#Score/Accuracy

print("Accuracy --> ", model.score(X_test, y_test)*100)
#Train the model

from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
#Fit

GBR.fit(X_train, y_train)
print("Accuracy --> ", GBR.score(X_test, y_test)*100)