#import the libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sb

sb.set(style="whitegrid", color_codes=True)

sb.set(font_scale=1)

from sklearn.ensemble import RandomForestRegressor

#import the dataset

trainset=pd.read_csv('../input/train.csv')

testset=pd.read_csv('../input/test.csv')
#display the top 5 rows of data

trainset.head()
#getting the overall view of data i.e no.of rows and columns

trainset.shape
#getting the datatype description of data

trainset.info()
# 35+3 continuous variables 43 categorical/object variables out of 81

trainset.get_dtype_counts()
#to get statistical dostribution of data

trainset.describe()
#to see how the data values are distributed 

trainset.hist(bins=20, figsize=(20,15))

plt.show()
#Using correlation matrix to know the important variable

trainset.corr()[trainset.corr() > 0.5]
corr=trainset.corr()['SalePrice']

corr[np.argsort(corr,axis=0)[::-1]]
#We can also use heatmap to get important variable

corr = trainset.corr()

fig, ax = plt.subplots(figsize=(30,30))

sb.heatmap(corr, annot=True, square=True, ax=ax, cmap='Blues')

plt.xticks(fontsize=20);

plt.yticks(fontsize=20);

# We see that Sales price is highly related with OverallQual, YearBuilt, YearRemodAdd, Total BsmSF, 

# TotalBsmtSF, 1stFlrSF, GrLivArea, FullBath, TotRmsAbvGrd, GarageCar, GarageArea as the correlation is greater than 0.5



corr_2 = trainset[['SalePrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',

                    '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']].corr()



fig, ax = plt.subplots(figsize=(15,15))

sb.heatmap(corr_2, annot=True, square=True, ax=ax, cmap='Blues')

plt.xticks(fontsize=10);

plt.yticks(fontsize=10);
#Visualize how your important variables are distributed 

from pandas.plotting import scatter_matrix

attributes = ['SalePrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',

             '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']



scatter_matrix(trainset[attributes], figsize=(20, 20));
# Categorical Variable compared with Target Variable like Overall 

trainset[['OverallQual','SalePrice']].groupby(['OverallQual'],

as_index=False).mean().sort_values(by='OverallQual', ascending=False)
# Checking which numbers are frequently occuring in a column



sb.distplot(trainset['SalePrice'], color="r", kde=False)

# sb.distplot(column, color, curve)

plt.title("Distribution of Sale Price")

plt.ylabel("Number of Occurences")

plt.xlabel("Sale Price");
#From bar graph we don't get proper idea of outliers so scatter plot is used 

plt.scatter(range(trainset.shape[0]), trainset["SalePrice"].values,color='orange')

plt.title("Distribution of Sale Price")

plt.xlabel("Number of Occurences")

plt.ylabel("Sale Price");
#We need to deal with outliers by setting up the upper limit and then clipping the data 

upperlimit = np.percentile(trainset.SalePrice.values, 99.5)

trainset['SalePrice'].loc[trainset['SalePrice']>upperlimit]=upperlimit

#upperlimit

plt.scatter(range(trainset.shape[0]), trainset["SalePrice"].values,color='orange')

plt.title("Distribution of Sale Price")

plt.xlabel("Number of Occurences")

plt.ylabel("Sale Price");
#Now we are going to handle the missing values 

#Plotting the no. of missing values in each column 

null_columns=trainset.columns[trainset.isnull().any()]

labels = []

values = []

for col in null_columns:

    labels.append(col)

    values.append(trainset[col].isnull().sum())

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,50))

ax.barh(ind, np.array(values), color='r')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_ylabel("Column Names")

ax.set_title("Variables with missing values");
# Counting missing values column-wise

missing_column = (trainset.isnull().sum())

print(missing_column[missing_column > 0])



#null_columns=houses.columns[houses.isnull().any()]

#houses[null_columns].isnull().sum()
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
# Predicting prices using random forest regressor

model = RandomForestRegressor()

model.fit(imputed_X_train,Y_train)

preds = model.predict(imputed_X_test)

print(preds)
preds.shape
#Output file

submission = pd.DataFrame({'Id': testset.Id, 'SalePrice': preds})

submission.to_csv('House_price_submission.csv', index=False)