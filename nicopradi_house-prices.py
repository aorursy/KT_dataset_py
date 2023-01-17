# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Graphs

import matplotlib.pyplot as plt

import seaborn as sns



from scipy import stats

from scipy.stats import norm



from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings('ignore')
# Get the training set

df_train = pd.read_csv('../input/train.csv')
# Check the column/features

df_train.columns
df_train['SalePrice'].describe()
# Histogram of the saleprice

sns.distplot(df_train['SalePrice']);

# Compute the 'skewness' and 'kurtosis'

print('Skewness: ', df_train['SalePrice'].skew())
# Plot scatter plots of expected variables having an influence on the price

# GrLivAre feature



data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)



# Scatter Plot

data.plot.scatter(x='GrLivArea', y='SalePrice')



## -> Linear relationship ?
# Same thing with the variable 'TotalBsmtSF'



data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis = 1)



# Scatter Plot

data.plot.scatter(x='TotalBsmtSF', y='SalePrice')



## -> Exponential relationship ? 
# Do the same thing with categorical features

# Feature: OverallQual



data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis = 1)



# MatplotLib

f, ax = plt.subplots(figsize=(8, 6)) # It make the plot bigger

# Use Seaborn

fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)



## Strong relationship
# Do the same thing with the feature 'YearBuilt'



data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis = 1)



# Matplotlib

f, ax = plt.subplots(figsize=(16, 8))



# SeaBorn

fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data)



## Not a strong dependency, new building tend to be more expensive than old ones

## Does sell price take into account the inflation ??
# Let's try to analyse the features in a more objective way

# 1/ Correlation matrix (heat map)

# 2/ 'SalePrice' Correlation matrx

# 3/ scatter plot between the most correlated variables



# 1/ Correlation matrix



correlation = df_train.corr() # Pair-wise correlation of columns, return a dataframe matrix



# Plot a heatmap

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(correlation)



# The last row is interesting to see which variable is correlated with 
# 2/ Correlation matrix for the output 'Saleprice'



k = 10 # Number of features to consider



# nlargest: Returns the first k rows ordered by 'SalePrice' in descending order

cols = correlation.nlargest(k, 'SalePrice')

cols = cols['SalePrice'] # Get a Serie (Keep only the Saleprice column)

cols = cols.index # Get the index of each row (The interesting features)



# corrcoef compute the correlation of a 2D array

top_correlation = np.corrcoef(df_train[cols].values.T)



# Plot a heatmap

sns.set(font_scale=1.25)

heat_map = sns.heatmap(top_correlation, cbar=True, annot=True, square=True, 

                      fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,

                      xticklabels=cols.values)

# 3/ scatter plot between the most correlated variables

# Let's make the scatter plots



sns.set() # Reset



# Features we want to plot against each other

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',

       'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size=2.5)
# Deal with the missing data

# -> How prevalent is the missing data ?

# -> Is missing data random, or does it have a pattern ?

# Can imply a reduction of the training set



boolean_missing = df_train.isnull() # Boolean dataframe (True if data is missing)

# Get the number of missing values per features (Series)

boolean_missing = boolean_missing.sum().sort_values(ascending=False) 



# Series containing the percentage of missing values per features

# the isnull() on the denominator is important because count() does not include N/A values

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)



# Summary of the missing data

missing_data = pd.concat([boolean_missing, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)



# If more than 15% of the data is missing, we should delete the feature (do not try to fill the missing value)

# Will we miss these features ? Unlikely

# 'GarageX' features data is missing at the same entries ? Anyway we'll only keep 'GarageCars' features

# Same for 'BsmtX' features
# So from the observation, delete all the features in the Series above,

# except for Eletrical for which we'll only delete the example with the missing value



# Delete features with a lot of missing entries

df_train = df_train.drop(missing_data.loc[missing_data['Total'] > 1].index, axis = 1)



# Delete the example with data missing in Eletrical

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# Check there is no missing values anymore

df_train.isnull().sum().max()
# Deal with outlier

# They may affect our models -> Valuable source of information, specific behaviours



# Analysis the standard deviation of 'SalePrice' and make some scatter plots

# Define a threshold that defines an observation as an outlier

# -> Mean normalization + Feature scaling necessary



# output an array

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])



# Look at the extrem values

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]



print('Low range:', low_range)

print('-'*40)

print('High range:', high_range)



# Fow low range values are not too far from 0

# For high range values, need to be carefull about the 5.0 and 7.0
# Try to analyse these 'outliers' based on the 'GrLivArea' feature



to_plot = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis = 1)



# Make the scatter plot

to_plot.plot.scatter(x = 'GrLivArea', y = 'SalePrice')
# The two dots on the right are strange, it could be agricultural area

# which would explain their low cost.

# We'll consider them as outliers and delete them



# The two dots on the top are the 7.x standard deviation operations.

# They seem follow the trend so we'll keep them



# Get the ID of the two outliers

df_train.sort_values(by = 'GrLivArea', ascending=False)[:2]

# Delete these two entries

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# What about relation between SalePrice and another interesting feature: 'TotalBsmtSF'



to_plot = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis = 1)



# Scatter plot

to_plot.plot.scatter(x='TotalBsmtSF', y='SalePrice')



# Some observations derive a bit from the trend, but let's keep every of them.
# Several statistics tests rely on assumption that the output variable, 

# 'SalePrice' is normally distributed



# Plot the histogram of SalePrice and compare to normal distribution

sns.distplot(df_train['SalePrice'], fit=norm)



# Probability Plot

fig = plt.figure()

# scipy plot

res = stats.probplot(df_train['SalePrice'], plot=plt)



# 'SalePrice' is not normal (peakedness) -> Data transformation needed

# Apply the log transformation
# Log transformation



df_train['SalePrice'] = np.log(df_train['SalePrice'])



# Recompute the previous graph to comapre SalePrice with the normal distribution

sns.distplot(df_train['SalePrice'], fit=norm) # -> Much better



# Probability plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt) # -> Much better
# Do the same thing with the other features:

# Check that their distribution is more or less gaussian



# 'GrLivArea'

sns.distplot(df_train['GrLivArea'], fit=norm)



# Probability Plot

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
# Feature to keep in the training set : 

#'SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',

# 'TotalBsmtSF', 'FullBath', 'YearBuilt'



features_to_keep = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'] 



# Create the training set for linear regression

X_train = df_train[features_to_keep]

Y_train = df_train['SalePrice']



X_train.shape, Y_train.shape
# Train a simple model: Logistic Regression



lin_reg = LinearRegression()

lin_reg.fit(X_train, Y_train)





# Compute the accuracy of the model on the training set (#correct predictions/#total predictions)

# LINEAR HYOPTHESIS

accu_lin_reg = round(lin_reg.score(X_train, Y_train) * 100, 2)

accu_lin_reg