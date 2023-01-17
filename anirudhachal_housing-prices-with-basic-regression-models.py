import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os



%matplotlib inline
# Loading test and training data

train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col = 'Id')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col = 'Id')
train_df
test_df
train_df.info()
test_df.info()
# Adding Sale Price Feature to test_df with default value -1

test_df['SalePrice'] = -1

test_df.info()
# Concatinating train_f and test_df to process them simultaneously

df = pd.concat((train_df, test_df), axis = 0)
df.info()
# Houses with invalid entries in 'MSZoning' feature

df[pd.isnull(df['MSZoning'])]



# We observe that all the houses with missing 'MSZoning' values are from the test dataset
# Frequency of different MSZoning Categories

df['MSZoning'].value_counts()
# Crosstab between 'MSSubClass' and 'MSZoning'

pd.crosstab(df['MSSubClass'], df['MSZoning'])
# Filling in missing MSZoning values



df.loc[pd.isnull(df['MSZoning']) & (df['MSSubClass'] == 20), 'MSZoning'] = 'RL'

df.MSZoning.fillna('RM', inplace = True)



#The above replacement is carried out because MSSubClass and MSZoning most probably are interrelated.
# MSZoning feature is now complete

df.info()
# Houses with invalid LotFrontage entries

df[pd.isnull(df['LotFrontage'])]
# Crosstab between MSSubClass and LotFrontage

pd.crosstab(df['MSSubClass'], df['LotFrontage'])
# Scatter Plot between MSSubClass and LotFrontage

plt.scatter(df['MSSubClass'], df['LotFrontage'], color = 'c', alpha = 0.2);



plt.title('Lot Frontage Vs MS SubClass')

plt.xlabel('MS SubClass')

plt.ylabel('Lot Frontage')



plt.show()
# Checking the Frequency of different Lot Frontage Values

df['LotFrontage'].value_counts().sort_index()
# Plot of frequency of LotFrontage values

plt.hist(df.loc[pd.notnull(df['LotFrontage'])]['LotFrontage'], bins = 80 ,color = 'c')



plt.title('LotFrontage Frequency Plot')

plt.xlabel('LotFrontage')

plt.ylabel('Frequency')



plt.show()
df.describe()
# Statistics of known LotFrontage Values

lot_frontage_mean = df.loc[pd.notnull(df['LotFrontage']), 'LotFrontage'].mean()

lot_frontage_median = df.loc[pd.notnull(df['LotFrontage']), 'LotFrontage'].median()

lot_frontage_std = df.loc[pd.notnull(df['LotFrontage']), 'LotFrontage'].std()



print(f"Lot Frontage mean : {lot_frontage_mean}") # Mean

print(f"Lot Frontage median : {lot_frontage_median}") # Median

print(f"Lot Frontage Standard Deviation : {lot_frontage_std}") # Standard Deviation
# Crosstab between MSZoning and LotFrontage

pd.crosstab(df['MSZoning'], df['LotFrontage'])
# Scatter Plot between MSZoning and LotFrontage

plt.scatter(df['MSZoning'], df['LotFrontage'], color = 'c', alpha = 0.2);



plt.title('Lot Frontage Vs MS Zoning')

plt.xlabel('MS Zoning')

plt.ylabel('Lot Frontage')



plt.show()
# Scatter Plot between LotArea and Lot Frontage

plt.scatter(df['LotArea'], df['LotFrontage'], color = 'c', alpha = 0.1);



plt.title('Lot Frontage vs Lot Area')

plt.xlabel('Lot Area')

plt.ylabel('Lot Frontage')



plt.show()
# Scatter Plot between LotArea and Lot Frontage Excluding Few Outlier from Lot Area

plt.scatter(df.loc[df['LotArea'] <= 50000, 'LotArea'], df.loc[df['LotArea'] <= 50000, 'LotFrontage'], color = 'c', alpha = 0.1);



plt.title('Lot Frontage vs Lot Area (Excluding outliers)')

plt.xlabel('Lot Area')

plt.ylabel('Lot Frontage')



plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



# Setting features matrix X with only one feature ie LotArea

X = df.loc[pd.notnull(df['LotFrontage']), 'LotArea'].to_numpy().astype('int64').reshape(-1, 1)



# Setting prediction matrix y with LotFrontage feature

y = df.loc[pd.notnull(df['LotFrontage']), 'LotFrontage'].ravel().astype('int64').reshape(-1, 1)



# Greating Linear Regression model

lr_1 = LinearRegression()



# Train model (with all valid training examples)

lr_1.fit(X, y)



# Checking the performance of the trained model

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    print('Score for linear regression model - version 1 - Trial {0} : {1:.2f}'.format(i + 1, lr_1.score(X_test, y_test)))



# Getting Predictions

prediction_y = lr_1.predict(X)



# Plot the Predictions

plt.scatter(df['LotArea'], df['LotFrontage'], color = 'c', alpha = 0.1);

plt.plot(X, prediction_y, color = 'k')



plt.title('Lot Frontage vs Lot Area')

plt.xlabel('Lot Area')

plt.ylabel('Lot Frontage')



plt.show()
# Setting features matrix X with only one feature ie LotArea without outliers

X = df.loc[pd.notnull(df['LotFrontage']) & (df['LotArea'] <= 25000), 'LotArea'].to_numpy().astype('int64').reshape(-1, 1)



# Setting prediction matrix y with LotFrontage feature without outliers

y = df.loc[pd.notnull(df['LotFrontage']) & (df['LotArea'] <= 25000), 'LotFrontage'].ravel().astype('int64').reshape(-1, 1)



# Greating Linear Regression model

lr_2 = LinearRegression()



# Train model (with all valid training examples and without outliers)

lr_2.fit(X, y)



# Checking the performance of the trained model

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    print('Score for linear regression model - version 2 - Trail {0} : {1:.2f}'.format(i + 1, lr_2.score(X_test, y_test)))



# Getting Predictions

prediction_y = lr_2.predict(X)

# Plot predictions

# Plot without outiers

plt.scatter(df.loc[df['LotArea'] <= 25000, 'LotArea'], df.loc[df['LotArea'] <= 25000, 'LotFrontage'], color = 'c', alpha = 0.1);

plt.plot(X, prediction_y, color = 'k');

plt.title('Lot Frontage vs Lot Area (Excluding outliers)')

plt.xlabel('Lot Area')

plt.ylabel('Lot Frontage')

plt.show()
# Plot with outliers (Model is still without outliers)

plt.scatter(df['LotArea'], df['LotFrontage'], color = 'c', alpha = 0.1);

plt.plot(X, prediction_y, color = 'k');

plt.title('Lot Frontage vs Lot Area')

plt.xlabel('Lot Area')

plt.ylabel('Lot Frontage')

plt.show()
# Checking out the number of outlying houses (wrt LotArea) having  missing Lot Frontage

df[pd.isnull(df['LotFrontage']) & (df['LotArea'] > 25000)]
# Setting these values to the median

outlier_median = df.loc[pd.notnull(df['LotFrontage']) & (df['LotArea'] > 25000), 'LotFrontage'].median()

df.loc[pd.isnull(df['LotFrontage']) & (df['LotArea'] > 25000), 'LotFrontage'] = outlier_median

print(f"Outlier Median : {outlier_median}")
# Outlier values have been filled in

df.info()
# Calculating predictions

X = df.loc[pd.isnull(df['LotFrontage']), 'LotArea'].to_numpy().astype('int64').reshape(-1, 1)

prediction_LotFrontage = lr_2.predict(X)



# Setting lot frontage values for houses with lot area <= 25000

df.loc[pd.isnull(df['LotFrontage']) & (df['LotArea'] <= 25000), 'LotFrontage'] = prediction_LotFrontage
# Checking to see the precitions fit the graph

# Scatter Plot between LotArea and Lot Frontage

plt.scatter(df['LotArea'], df['LotFrontage'], color = 'c', alpha = 0.1);



plt.title('Lot Frontage vs Lot Area (After filling in missing values)')

plt.xlabel('Lot Area')

plt.ylabel('Lot Frontage')



plt.show()
# Lot Frontage value is now complete

df.info()
# Houses with missing values in this feature

df[pd.isnull(df['Alley'])]
df[pd.notnull(df['Alley'])]
# Crosstab between Street and Alley

pd.crosstab(df['Street'], df['Alley'])
# Setting missing values to NA

df.loc[pd.isnull(df['Alley']), 'Alley'] = 'NA'
# Checking if update done as expected

df['Alley'].value_counts()
# Alley feature is now complete

df.info()
# Checking out houses with missing Utilities

df[pd.isnull(df['Utilities'])]
# Observing frequency of different Utilities entries

df['Utilities'].value_counts()
# Observing frequency of different Utilities entries (normalized)

df['Utilities'].value_counts(normalize = True) * 100
# Setting missing values to 'AllPub'

df.loc[pd.isnull(df['Utilities']), 'Utilities'] = 'AllPub'



# Checking to see update done properly

df['Utilities'].value_counts()
# Utilities feature is now ready

df.info()
# Checking houses with missing Exterior Covering values

df.loc[pd.isnull(df['Exterior1st']) | pd.isnull(df['Exterior2nd'])]
# Checking the frequecy of different housing exteriors

print(df['Exterior1st'].value_counts(), "\n")

print(df['Exterior2nd'].value_counts())
# Checking to see which year this house was built

df.loc[pd.isnull(df['Exterior1st']) | pd.isnull(df['Exterior2nd']), 'YearBuilt']
# Checking to see if there is a relationship between these features

print(df.loc[df['YearBuilt'] == 1940, 'Exterior1st'].value_counts(), "\n")

print(df.loc[df['YearBuilt'] == 1940, 'Exterior2nd'].value_counts())
# Checking to see the if there is any relationship between class of the house and the type of exteriors used

print(df.loc[df['MSSubClass'] == 30, 'Exterior1st'].value_counts(), "\n")

print(df.loc[df['MSSubClass'] == 30, 'Exterior2nd'].value_counts())
# Checking to see which year this house was remodeled 

df.loc[pd.isnull(df['Exterior1st']) | pd.isnull(df['Exterior2nd']), 'YearRemodAdd']
# Checking to see if there is a relationship between these features

print(df.loc[df['YearRemodAdd'] >= 2005, 'Exterior1st'].value_counts(), "\n")

print(df.loc[df['YearRemodAdd'] >= 2005, 'Exterior2nd'].value_counts())
# Checking to see if there is a relationship between these features (combining features)

print(df.loc[(df['YearRemodAdd'] >= 2005) & (df['MSSubClass'] == 30), 'Exterior1st'].value_counts(), "\n")

print(df.loc[(df['YearRemodAdd'] >= 2005) & (df['MSSubClass'] == 30), 'Exterior2nd'].value_counts())
# Setting housing exterior (both 1st and 2nd) to Ws Sdng

df.loc[pd.isnull(df['Exterior1st']), ['Exterior1st', 'Exterior2nd']] = 'Wd Sdng'
# Checking to see update done correctly

print(df['Exterior1st'].value_counts(), "\n")

print(df['Exterior2nd'].value_counts())
# Housing Exterior feature is now complete

df.info()
# Housing with missing Masonary Veneer Type feature

df[pd.isnull(df['MasVnrType'])]
# Distribution of different entries

df['MasVnrType'].value_counts()
# Crosstab between MSSubClass and MasVnrType

pd.crosstab(df['MSSubClass'], df['MasVnrType'])
# Crosstab between Housing Exterior and MasVnrType

pd.crosstab(df['Exterior1st'], df['MasVnrType'])
pd.crosstab(df.loc[df['Exterior1st'] == 'HdBoard', 'MSSubClass'], df.loc[df['Exterior1st'] == 'HdBoard','MasVnrType'])
df[pd.isnull(df['MasVnrType']) & (df['Exterior1st'] == 'HdBoard')]
# Filling in missing values

df['MasVnrType'].fillna('None', inplace = True)
# Checking to see if the update was done correctly

df['MasVnrType'].value_counts()
# MasVnrType feature is now complete

df.info()
df['MasVnrArea'].value_counts()
# Plot the frequency of the Masonary Veneer Area

plt.hist(df.loc[pd.notnull(df['MasVnrArea']), 'MasVnrArea'], bins = 35, color = 'c')



plt.title('Distrubution of Masonary Veneer Area')

plt.xlabel('Masonary Veneer Area')

plt.ylabel('Frequency')



plt.show()
# Setting these missing values to be 0

df.loc[(df['MasVnrType'] == 'None') & (pd.isnull(df['MasVnrArea'])), 'MasVnrArea'] = 0
df.loc[(pd.isnull(df['MasVnrArea']))]
df['MasVnrArea']
# MasVnrArea is now complete

df.info()
# Houses with missing values in features related to basement

df[pd.isnull(df['BsmtQual'])]
df[pd.isnull(df['BsmtCond'])]
df[pd.isnull(df['BsmtExposure'])]
df[pd.isnull(df['BsmtFinType1'])]
df[pd.isnull(df['BsmtFinType2'])]
is_one_basement_missing = pd.isnull(df['BsmtQual']) | pd.isnull(df['BsmtCond']) | pd.isnull(df['BsmtExposure']) | pd.isnull(df['BsmtFinType1']) | pd.isnull(df['BsmtFinType2'])

df[is_one_basement_missing]
is_all_basement_missing = pd.isnull(df['BsmtQual']) & pd.isnull(df['BsmtCond']) & pd.isnull(df['BsmtExposure']) & pd.isnull(df['BsmtFinType1']) & pd.isnull(df['BsmtFinType2'])

df[is_all_basement_missing]
# Distribution of different entries

for column in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:

    print(df[column].value_counts(), "\n")

    
df['BsmtFinSF1'].value_counts()
# Houses with basement area 0 and invalid entries.

df.loc[(df['BsmtFinSF1'] == 0) & is_all_basement_missing, 'BsmtQual':]
# Setting all these to NA (no basement)

df.loc[(df['BsmtFinSF1'] == 0) & is_all_basement_missing, ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'NA'
# Checking to see if update was done as expected

df.loc[(df['BsmtFinSF1'] == 0) & is_all_basement_missing, 'BsmtQual':]
# Checking new distribution

for column in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:

    print(df[column].value_counts(), "\n")

    
# Update was done as expected 

df.info()
# Houses with basement area 0 and invalid entries.

is_one_basement_missing = pd.isnull(df['BsmtQual']) | pd.isnull(df['BsmtCond']) | pd.isnull(df['BsmtExposure']) | pd.isnull(df['BsmtFinType1']) | pd.isnull(df['BsmtFinType2'])

df.loc[(df['BsmtFinSF1'] == 0) & is_one_basement_missing, 'BsmtQual':]
df.loc[is_one_basement_missing, 'BsmtQual':]
df.loc[(df['BsmtQual'] == 'Gd'), 'BsmtCond'].value_counts()
# Setting housing with Good basement quality to TA

df.loc[(df['BsmtQual'] == 'Gd') & pd.isnull(df['BsmtCond']), 'BsmtCond'] = 'TA'
is_one_basement_missing = pd.isnull(df['BsmtQual']) | pd.isnull(df['BsmtCond']) | pd.isnull(df['BsmtExposure']) | pd.isnull(df['BsmtFinType1']) | pd.isnull(df['BsmtFinType2'])

df.loc[is_one_basement_missing, 'BsmtQual':]
# Checking houses with good basement height and an avergage condition

df.loc[(df['BsmtQual'] == 'Gd') & (df['BsmtCond'] == 'TA'), 'BsmtExposure'].value_counts()
# Setting these missing values to No as that is the entry most houses of this type have

df.loc[(df['BsmtQual'] == 'Gd') & (df['BsmtCond'] == 'TA') & pd.isnull(df['BsmtExposure']), 'BsmtExposure'] = 'No'
is_one_basement_missing = pd.isnull(df['BsmtQual']) | pd.isnull(df['BsmtCond']) | pd.isnull(df['BsmtExposure']) | pd.isnull(df['BsmtFinType1']) | pd.isnull(df['BsmtFinType2'])

df.loc[is_one_basement_missing, 'BsmtQual':]
# Checking out houses with typical basement height

df.loc[(df['BsmtQual'] == 'TA'), 'BsmtCond'].value_counts()
# Setting these missing values to TA as the above distribution shows a clear bias

df.loc[(df['BsmtQual'] == 'TA')  & pd.isnull(df['BsmtCond']), 'BsmtCond'] = 'TA'
is_one_basement_missing = pd.isnull(df['BsmtQual']) | pd.isnull(df['BsmtCond']) | pd.isnull(df['BsmtExposure']) | pd.isnull(df['BsmtFinType1']) | pd.isnull(df['BsmtFinType2'])

df.loc[is_one_basement_missing, 'BsmtQual':]
df.loc[pd.isnull(df['TotalBsmtSF']), ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'NA'
df.loc[pd.isnull(df['TotalBsmtSF']), 'BsmtQual':]
df.loc[pd.isnull(df['TotalBsmtSF']), ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtUnfSF']]
df.loc[pd.isnull(df['TotalBsmtSF']), ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtUnfSF']] = 0

df.info()
is_one_basement_missing = pd.isnull(df['BsmtQual']) | pd.isnull(df['BsmtCond']) | pd.isnull(df['BsmtExposure']) | pd.isnull(df['BsmtFinType1']) | pd.isnull(df['BsmtFinType2'])

df.loc[is_one_basement_missing, 'BsmtQual':]
df.loc[df['BsmtCond'] == 'TA', 'BsmtQual'].value_counts()
df.loc[(df['BsmtCond'] == 'TA') & (df['BsmtExposure'] == 'No'), 'BsmtQual'].value_counts()
# Setting it to be TA since that seems to be the best choice

df.loc[pd.isnull(df['BsmtQual']) & (df['BsmtCond'] == 'TA'), 'BsmtQual'] = 'TA'
is_one_basement_missing = pd.isnull(df['BsmtQual']) | pd.isnull(df['BsmtCond']) | pd.isnull(df['BsmtExposure']) | pd.isnull(df['BsmtFinType1']) | pd.isnull(df['BsmtFinType2'])

df.loc[is_one_basement_missing, 'BsmtQual':]
df.loc[df['BsmtFinType1'] == 'GLQ', 'BsmtFinType2'].value_counts()
types = df['BsmtFinType1'].unique()



for _type in types:

    if _type == 'Unf' or _type == 'NA' or _type == 'TA' or _type is np.nan:

        continue

    plt.hist(df.loc[(df['BsmtFinType1'] == _type), 'BsmtFinSF1'], color = 'c', bins = 50)

    plt.xlabel(f'Surface area of finish type : {_type}')

    plt.ylabel('Frequency')

    plt.title(f'Frquency of Surface areas of type : {_type}')

    plt.tight_layout()

    plt.show()
types = df['BsmtFinType1'].unique()



for _type in types:

    if _type == 'Unf' or _type == 'NA' or _type == 'TA' or _type is np.nan:

        continue

    plt.hist(df.loc[(df['BsmtFinType1'] == _type) & (df['BsmtFinSF1'] < 600) & (df['BsmtFinSF1'] > 400), 'BsmtFinSF1'], color = 'c', bins = 10)

    plt.xlabel(f'Surface area of finish type : {_type}')

    plt.ylabel('Frequency')

    plt.title(f'Frquency of Surface areas of type : {_type}')

    plt.tight_layout()

    plt.show()
types = df['BsmtFinType1'].unique()

for _type in types:

    if _type == 'Unf' or _type == 'NA' or _type == 'TA' or _type is np.nan:

        continue

        

    count = df.loc[(df['BsmtFinType1'] == _type) & (df['BsmtFinSF1'] < 600) & (df['BsmtFinSF1'] > 400), 'BsmtFinSF1'].value_counts().sum()

    print(f"{_type} count : {count}")
# Setting value to ALQ

df.loc[pd.isnull(df['BsmtFinType2']), 'BsmtFinType2'] = 'ALQ'
is_one_basement_missing = pd.isnull(df['BsmtQual']) | pd.isnull(df['BsmtCond']) | pd.isnull(df['BsmtExposure']) | pd.isnull(df['BsmtFinType1']) | pd.isnull(df['BsmtFinType2'])

df.loc[is_one_basement_missing, 'BsmtQual':]
df.loc[df['BsmtCond'] == 'Fa', 'BsmtQual'].value_counts()
# Setting it to TA as that is the most common occuring entry

df.loc[(df['BsmtCond'] == 'Fa') & pd.isnull(df['BsmtQual']), 'BsmtQual'] = 'TA'
df.info()
# House with missing TotalBsmtSF value

df.loc[pd.isnull(df['TotalBsmtSF']), 'BsmtQual' :]
# Setting it to 0 as there is no basement

df.loc[pd.isnull(df['TotalBsmtSF']), 'TotalBsmtSF'] = 0
# Checking to make sure nothing is abonormal

df.loc[df['BsmtQual'] == 'NA', 'TotalBsmtSF'].value_counts()
# All Basement features are now ready

df.info()
# House with missing value in electrical feature

df[pd.isnull(df['Electrical'])]
# Distribution of entries

df['Electrical'].value_counts()
# Setting the value to SBrkr since the house has a relatively high class and also since that is the most common entry

df.loc[pd.isnull(df['Electrical']), 'Electrical'] = 'SBrkr'
# Checking if update done correctly

df['Electrical'].value_counts()
# Electrical Feature is now complete

df.info()
# House with missing basement full bathroom values

df.loc[pd.isnull(df['BsmtFullBath']), 'BsmtQual':]
# House with missing basement half bathroom values

df.loc[pd.isnull(df['BsmtHalfBath']), 'BsmtQual':]
# Setting values to 0

df.loc[pd.isnull(df['BsmtFullBath']), 'BsmtFullBath'] = 0

df.loc[pd.isnull(df['BsmtHalfBath']), 'BsmtHalfBath'] = 0
# Checking to see if there are any houses whose entires are impossible

df.loc[(df['BsmtFullBath'] != 0) & (df['BsmtQual'] == 'NA')]
# Basement Bathroom count feature is now complete

df.info()
# Houses with ivalid entry in KitchenQual feature

df.loc[pd.isnull(df['KitchenQual']), 'KitchenAbvGr':]
# Houses with ivalid entry in KitchenQual feature

df.loc[pd.isnull(df['KitchenQual'])]
# Distribution of KitchenQual

df['KitchenQual'].value_counts()
# Distribution of KitchenAbvGr

df['KitchenAbvGr'].value_counts()
# KitchenQual of houses with Kitchen above grade 1

df.loc[df['KitchenAbvGr'] == 1, 'KitchenQual'].value_counts()
# KitchenQual of houses with Kitchen above grade 1 and class is 50

df.loc[(df['KitchenAbvGr'] == 1) & (df['MSSubClass'] == 50), 'KitchenQual'].value_counts()
# Houses with ivalid entry in KitchenQual feature

df.loc[pd.isnull(df['KitchenQual']), 'KitchenQual'] = 'TA'
# Checking to see update done correctly

df['KitchenQual'].value_counts()
# Kitchen features are now complete

df.info()
# Houses with invalid entry in Functional feature

df[pd.isnull(df['Functional'])]
# Distribution of this feature

df['Functional'].value_counts()
# Setting missing values to typical by default

df.loc[pd.isnull(df['Functional']), 'Functional'] = 'Typ'
# Checkig if update done as expected

df['Functional'].value_counts()
# Functional Feature is now complete

df.info()
# Houses with missing values in the FireplaceQu feature

df[pd.isnull(df['FireplaceQu'])]
df.loc[pd.isnull(df['FireplaceQu']), 'Fireplaces':]
# Distribution of entries in FireplaceQu feature

df['FireplaceQu'].value_counts()
# Distrubtion of number of fireplaces for houses with missing FireplaceQu values

df.loc[pd.isnull(df['FireplaceQu']), 'Fireplaces'].value_counts()
# Setting missing values to 'NA'

df.loc[pd.isnull(df['FireplaceQu']), 'FireplaceQu'] = 'NA'
# Checking if update done correclty

df['FireplaceQu'].value_counts()
# FireplaceQu Feature now complete

df.info()
# Checking houses with garage area 0

df[df['GarageArea'] == 0]
# Checking houses with garage area 0

df.loc[df['GarageArea'] == 0, "GarageType" :]
# Checking distrbution of entries in garage features for houses with garage area 0

garage_features = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']



for feature in garage_features:

    print("Uniques in :" , df.loc[(df['GarageArea'] == 0) , feature].unique(), "\n")
# Checking distrubutions before update



for feature in garage_features:

    print(df[feature].value_counts(), "\n")

    
# Setting all garage related features of houses with garage area 0 to NA except for year built

# I will set year built feature to 0 since it was never built



for feature in garage_features:

    if feature != 'GarageYrBlt':

        df.loc[(df['GarageArea'] == 0) , feature] = 'NA'

    else :

        df.loc[(df['GarageArea'] == 0) , feature] = 0
# Checking if update done correctly

for feature in garage_features:

    print("Uniques in :" , df.loc[(df['GarageArea'] == 0) , feature].unique(), "\n")

    

for feature in garage_features:

    print(df[feature].value_counts(), "\n")

    
df.info()
# Checking out remaining houses

df.loc[pd.isnull(df['GarageYrBlt']), 'GarageType':]
# Checking out distribution of years in which detached garages were built

df.loc[df['GarageType'] == 'Detchd', 'GarageYrBlt'].value_counts(sort = True)
# Checking the frequency of years in which detached garages were built

plt.hist(df.loc[pd.notnull(df['GarageYrBlt']) & (df['GarageType'] == 'Detchd'), 'GarageYrBlt'], color = 'c', bins = 30);
median_year = df.loc[pd.notnull(df['GarageYrBlt']), 'GarageYrBlt'].median()
# Setting value to median

df.loc[pd.isnull(df['GarageYrBlt']), 'GarageYrBlt'] = median_year
# Checking if update done correctly

df.loc[pd.isnull(df['GarageYrBlt']), 'GarageType':]
# Houses with missing GarageFinishes

df.loc[pd.isnull(df['GarageFinish']), 'GarageType':]
df.loc[df['GarageType'] == 'Detchd', 'GarageFinish'].value_counts()
# Setting finishing to Unfinished as that is the most common entry

df.loc[pd.isnull(df['GarageFinish']), 'GarageFinish'] = 'Unf'
# Checking if the update was done correctly

df.loc[df['GarageType'] == 'Detchd', 'GarageFinish'].value_counts()
# Houses with missing GarageFinishes

df.loc[pd.isnull(df['GarageCars']), 'GarageType' : ]
# Distribution of GarageCars entries

df['GarageCars'].value_counts()
# Distribution of GarageCars entries among detached garages

df.loc[df['GarageType'] == 'Detchd','GarageCars'].value_counts()
# Distribution of GarageCars entries among detached garages built before 1985

df.loc[(df['GarageType'] == 'Detchd') & (df['GarageYrBlt'] < 1985),'GarageCars'].value_counts()
# Setting car capicity to 1

df.loc[pd.isnull(df['GarageCars']), 'GarageCars'] = 1
# Checking if update done as expected

df['GarageCars'].value_counts()
# Houses with missing Garage Area values

df.loc[pd.isnull(df['GarageArea']), 'GarageType' : ]
median_garage_size = df.loc[(df['GarageType'] == 'Detchd') & (df['GarageYrBlt'] < 1985) & (df['GarageCars'] == 1), 'GarageArea'].median()

print(f"Median garage size : {median_garage_size}")
# Setting missing value to this median value

df.loc[pd.isnull(df['GarageArea']), 'GarageArea'] = median_garage_size
# Houses with missing Garage Qual values

df.loc[pd.isnull(df['GarageQual']), 'GarageType' :]
# Checking distribution of quality of garages of houses having garage of similar type

df.loc[(df['GarageType'] == 'Detchd') & (df['GarageYrBlt'] < 1985) & (df['GarageCars'] == 1), 'GarageQual'].value_counts()
# Setting quality to 'TA' as suggested by the distribution

df.loc[pd.isnull(df['GarageQual']), 'GarageQual'] = 'TA'
# Checking distribution of condition of garages of houses having garage of similar type

df.loc[(df['GarageType'] == 'Detchd') & (df['GarageYrBlt'] < 1985) & (df['GarageCars'] == 1), 'GarageCond'].value_counts()
# Setting Condition to 'TA' as suggested by the distribution

df.loc[pd.isnull(df['GarageCond']), 'GarageCond'] = 'TA'
# All garage features are now complete

df.info()
# Housing with missing values in this feature

df.loc[pd.isnull(df['PoolQC']), 'PoolArea':]
df.loc[(pd.isnull(df['PoolQC'])) & (df['PoolArea'] == 0), 'PoolArea':]
# Setting value to NA

df.loc[(pd.isnull(df['PoolQC'])) & (df['PoolArea'] == 0), 'PoolQC'] = 'NA'
# Remainnig Houses

df.loc[pd.isnull(df['PoolQC']), 'PoolArea':]
#All houses having pools

df.loc[df['PoolArea'] != 0, 'PoolArea':]
df.loc[df['PoolArea'] != 0, 'PoolQC'].value_counts()
# Setting default pool quality to be good

df.loc[pd.isnull(df['PoolQC']), 'PoolQC'] = 'Gd'
# Pool feature is now complete

df.info()
# Houses with missing values in fence feature

df.loc[pd.isnull(df['Fence']), 'PoolArea':]
# Distribution of entries in fence feature

df['Fence'].value_counts()
# Setting default value to NA

df.loc[pd.isnull(df['Fence']), 'Fence'] = 'NA'
# Checking if update done as expected

df['Fence'].value_counts()
# Fence feature is now complete

df.info()
# Houses with missing values in this feature

df.loc[pd.isnull(df['MiscFeature'])]
# Distribution of entries

df['MiscFeature'].value_counts()
# Setting default value to NA

df.loc[pd.isnull(df['MiscFeature']), 'MiscFeature'] = 'NA'
# Checking if update done as expected

df['MiscFeature'].value_counts()
# This feature is now complete

df.info()
# House with missing value in the feature

df[pd.isnull(df['SaleType'])]
df.loc[pd.isnull(df['SaleType']), 'GarageType':]
# Distrubtion of entries

df['SaleType'].value_counts()
# Distrbution of entries of house solve after 2000

df.loc[df['YrSold'] == 2007, 'SaleType'].value_counts()
# Setting value to 'WD' since that is the most common entry

df.loc[pd.isnull(df['SaleType']), 'SaleType'] = 'WD'
# Checking to see that the update is done correctly

df['SaleType'].value_counts()
# This features is now complete

df.info()
# Identifying ordinal columns and assigning weights to them



ordinal_columns = [

    'LotShape', 

    'LandContour', 

    'Utilities', 

    'LandSlope', 

    'ExterQual', 

    'ExterCond', 

    'BsmtQual', 

    'BsmtCond',

    'BsmtExposure',

    'BsmtFinType1',

    'BsmtFinType2',

    'HeatingQC',

    'CentralAir',

    'Electrical',

    'KitchenQual',

    'Functional',

    'FireplaceQu',

    'GarageFinish',

    'GarageQual',

    'GarageCond',

    'PoolQC'  

]



ordinal_columns_count = len(ordinal_columns)

print(f'Oridinal column count : {ordinal_columns_count}')



weights = {

    # LotShape

    'Reg' : 4,

    'IR1' : 3,

    'IR2' : 2,

    'IR3' : 1,

    

    # LandContour

    'Lvl' : 4,

    'Bnk' : 3,

    'HLS' : 2,

    'Low' : 1,

    

    # Utilities

    'AllPub' : 4,

    'NoSewr' : 3,

    'NoSeWa' : 2,

    'ELO' : 1,

    

    # LandSlope

    'Gtl' : 1,

    'Mod' : 2,

    'Sev' : 3,

    

    # ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond,

    # PoolQC, 

    'Ex' : 5,

    'Gd' : 4,

    'TA' : 3,

    'Fa' : 2,

    'Po' : 1,

    'NA' : 0,

    

    # BsmtExposure

    'Av' : 3,

    'Mn' : 2,

    'No' : 1,

    

    # BsmtFinType1, BsmtFinType2

    'GLQ' : 6,

    'ALQ' : 5,

    'BLQ' : 4,

    'Rec' : 3,

    'LwQ' : 2,

    'Unf' : 1,

    

    # CentralAir

    'N' : 0,

    'Y' : 1,

    

    # Electrical

    'SBrkr' : 4,

    'FuseA' : 3,

    'FuseF' : 2, 

    'FuseP' : 1, 

    'Mix' : 3,

    

    # Functional

    'Typ' : 8,

    'Min1' : 7,

    'Min2' : 6,

    'Mod' : 5,

    'Maj1' : 4, 

    'Maj2' : 3, 

    'Sev' : 2,

    'Sal' : 1,

    

    # GarageFinish

    'Fin' : 3,

    'RFn' : 2,

    'Unf' : 1

    

}
# Setting all the values of the ordinal columns in the df to their numeric weightage

for column in ordinal_columns:

    df[column] = df[column].map(weights)

df[ordinal_columns]
df.info()
# Getting dummy columns for all object columns

dummy_columns = []

numerical_columns_count = 0

for column in df.columns:

    if df[column].dtype == 'object':

        dummy_columns.append(column)

    else :

        numerical_columns_count += 1

dummy_columns
# No of unique new features that we will get

dummy_columns_count = 0

for column in dummy_columns:

    dummy_columns_count += len(df[column].unique())

print(f'dummy columns count : {dummy_columns_count}')
# Getting Dummies

df = pd.get_dummies(df, dummy_columns)
# reorder columms

columns = [column for column in df.columns if column != 'SalePrice']

columns = ['SalePrice'] + columns

df = df[columns]

df.info(verbose = True)
# Checking if everthing is alright

df.head()
df
df.info(verbose = False)
# Checking to see that we have not lost any features

if numerical_columns_count + dummy_columns_count == len(df.columns):

    print("Process successfully completed!")

else:

    print("Error...")
# Separating the training and test parts

train_df = df[df['SalePrice'] != -1]

columns = [column for column in columns if column != 'SalePrice']

test_df = df.loc[df['SalePrice'] == -1, columns]
train_df
test_df
# Function to make prediction files

def get_prediction_file(model, filename):

    # Making predictions

    test_X = test_df.loc[:, :].to_numpy().astype('float')

    predictions_y = model.predict(test_X) 

    

    # Setting up submission dataframe

    df_submission = pd.DataFrame({'Id' : test_df.index, 'SalePrice' : predictions_y})

    

    # Setting path

    submission_data_path = os.path.join(os.path.pardir, 'Data', 'Predictions')

    submission_file_path = os.path.join(submission_data_path, filename)

    # write to the file

    df_submission.to_csv(submission_file_path, index = False)
# Splitting into test train group

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error



X = train_df.loc[:, 'MSSubClass':].to_numpy().astype('float')

y = train_df.loc[:, 'SalePrice'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Function to test the performance of a model

def test_model(model, count, X_temp, y_temp):

    for i in range(count):

        # Test train split

        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size = 0.2)

        

        # Fitting the model

        model.fit(X_train_temp, y_train_temp)

        

        # Making predictions and printing error

        y_predictions = model.predict(X_test_temp)

        print(f"Mean Square error of model Trial {i + 1}: ", mean_squared_error(y_predictions, y_test_temp, squared = True))

        if (y_predictions < 0).any() or (y_test_temp < 0).any():

            continue

        print(f"Mean Square log error of model Trial {i + 1}: ", mean_squared_log_error(y_predictions, y_test_temp))

        
# BaseLine Model 

from sklearn.dummy import DummyRegressor



dummy_model = DummyRegressor()



test_model(dummy_model, 10, X, y)
# Building first linear regression model

from sklearn.linear_model import LinearRegression



lr_1 = LinearRegression()



test_model(lr_1, 10, X, y)
# Getting submssion file for linear regression model



lr_1.fit(X, y)



X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size = 0.2)

y_predictions = lr_1.predict(X_test_temp)

print(f"Mean Square error of model Trial {i + 1}: ", mean_squared_error(y_predictions, y_test_temp))



# get_prediction_file(lr_1, 'lr_1.1.csv')
# Trying out polynomial regression

from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(2)

X_poly = poly.fit_transform(X)



# Applying Regression on this new matrix

lr_2 = LinearRegression()



test_model(lr_2, 10, X_poly, y)
# Trying Random Forest Classifier

from sklearn.ensemble import RandomForestRegressor



RFR = RandomForestRegressor(max_depth=3, random_state=0)



test_model(RFR, 10, X, y)
# Getting prediction file using this model

RFR.fit(X, y)



X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size = 0.2)

y_predictions = RFR.predict(X_test_temp)

print(f"Mean Square error of model Trial {i + 1}: ", mean_squared_error(y_predictions, y_test_temp))

print(f"Mean Square log error of model Trial {i + 1}: ", mean_squared_log_error(y_predictions, y_test_temp))



# get_prediction_file(RFR, 'RFR_1.0.csv')