# Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style = 'whitegrid', color_codes = True, font_scale = 1)

from sklearn.ensemble import RandomForestRegressor
trainset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

testset = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
trainset.head()
testset.head()
trainset.info()
trainset.get_dtype_counts()
trainset.describe()
trainset.hist(bins = 20, figsize = (20,10))

plt.show()
# Let's use correlation matrix to know important variable

trainset.corr()[trainset.corr() > 0.5]
corr = trainset.corr()['SalePrice']

corr[np.argsort(corr, axis = 0)[::-1]]
# We can use heatmap to get to know important variables

corr = trainset.corr()

fig, ax = plt.subplots(figsize = (30,30))

sns.heatmap(corr, annot = True, square = True, ax = ax, cmap = 'Greens')

plt.xticks(fontsize = 20);

plt.yticks(fontsize = 20);
# We see that Sales price is highly related with OverallQual, YearBuilt, YearRemodAdd, Total BsmSF, 

# TotalBsmtSF, 1stFlrSF, GrLivArea, FullBath, TotRmsAbvGrd, GarageCar, GarageArea as the correlation is greater than 0.5

corr2 = trainset[['SalePrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF','GrLivArea', 

                  'FullBath','TotRmsAbvGrd','GarageCars','GarageArea']].corr()



fig, ax = plt.subplots(figsize = (15,15))

sns.heatmap(corr2, annot = True, square = True, ax = ax, cmap = "Greens")

plt.xticks(fontsize = 10);

plt.yticks(fontsize = 10);
# Visulatizing our final variables

from pandas.plotting import scatter_matrix

attributes = ['SalePrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF','GrLivArea', 

                  'FullBath','TotRmsAbvGrd','GarageCars','GarageArea']

scatter_matrix(trainset[attributes], figsize = (20,20));
trainset[['OverallQual', 'SalePrice']].groupby(['OverallQual'],

                        as_index = False).mean().sort_values(by = 'OverallQual', ascending = False)

# Checking wwhich numbers are frequently occuring

sns.distplot(trainset['SalePrice'], color = 'g', kde = False)

plt.title('Distribution of sale')

plt.xlabel('Number of occurance')

plt.ylabel('Sale Price')
# Let's use scatter plot for better view

plt.scatter(range(trainset.shape[0]), trainset['SalePrice'].values, color = 'pink' )

plt.title('Distribution of sale')

plt.xlabel('Number of occurance')

plt.ylabel('Sale Price')
# We will deal with outliers by using upper limit

upperlimit = np.percentile(trainset.SalePrice.values, 99.5)

trainset['SalePrice'].loc[trainset['SalePrice'] > upperlimit] = upperlimit

plt.scatter(range(trainset.shape[0]), trainset['SalePrice'].values, color = 'orange')

plt.title('Distribution of sale')

plt.xlabel('Number of occurance')

plt.ylabel('Sale Price')
null_columns = trainset.columns[trainset.isnull().any()]

labels = []

values = []



for col in null_columns:

    labels.append(col)

    values.append(trainset[col].isnull().sum())

ind = np.arange(len(labels))    

widths = 0.9

fig, ax = plt.subplots(figsize = (20,50))

ax.barh(ind, np.array(values), color = 'r')

ax.set_xticks(ind +((widths)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_ylabel("Column Names")

ax.set_title("Variables with missing values");
missing_columns = (trainset.isnull().sum())

print(missing_columns[missing_columns > 0])
#Forming the train set with only continuous variables 

X_train = trainset.drop('SalePrice',axis=1)

X_train = X_train.select_dtypes(exclude=['object'])

Y_train = trainset.SalePrice

X_test = testset.select_dtypes(exclude=['object'])
#Imputing the missing value and keeping the columns 

imputed_X_train = X_train.copy()

imputed_X_test = X_test.copy()

# Copying the orginal data,original data should not change(avoid it)

col_missing_val = (col for col in X_train.columns if X_train[col].isnull().any())

# Any column having missing values, it will be put into above variable

for col in col_missing_val:

    imputed_X_train[col +'_was_missing'] = imputed_X_train[col].isnull()

    imputed_X_test[col +'_was_missing'] = imputed_X_test[col].isnull()

#Imputer

from sklearn.preprocessing import Imputer

my_imputer =Imputer()

imputed_X_train = my_imputer.fit_transform(imputed_X_train)

imputed_X_test = my_imputer.transform(imputed_X_test)
# Predicting the model

model = RandomForestRegressor()

model.fit(imputed_X_train, Y_train)

pres = model.predict(imputed_X_test)

print(pres)
pres.shape
submission = pd.DataFrame({'Id' : testset.Id, 'SalePrice' : pres})

submission.to_csv('House_Price_Submisson', index= False)