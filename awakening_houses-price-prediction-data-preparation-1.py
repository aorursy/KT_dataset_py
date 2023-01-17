import numpy as np
import pandas as pd
import numpy.random as nr
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn import preprocessing
from sklearn import linear_model

%matplotlib inline
hp_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
hp_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
hp_train.shape
hp_test.shape
hp_train.columns
hp_test.columns
hp_train.head(2)
hp_test.head(2)
hp_train.index
# hp_train.dtypes
# The SalePrice Column is not Present in the Test Data, So we Add it as Null Value

hp_test ['SalePrice'] = 'NaN'
house_prices = pd.concat([hp_train, hp_test]).reset_index(drop = True)
house_prices.shape
house_prices.columns
# Setting the Rows and Columns to Display all Values without Compressing

pd.options.display.max_columns = None
pd.options.display.max_rows = None
house_prices.head(2)
house_prices.tail(2)
house_prices.index
house_prices.dtypes
# Differentiate numerical features and Categorical Features

categorical_features = house_prices.select_dtypes(include = ["object"]).columns
numerical_features = house_prices.select_dtypes(exclude = ["object"]).columns
categorical_features = categorical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
house_pricesnum = house_prices[numerical_features]
house_pricescat = house_prices[categorical_features]
numerical_features
categorical_features
# house_prices['SalePrice'] = pd.to_numeric(house_prices['SalePrice'])
house_prices.describe()
# Identify the Categorical Columns

cat_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
            'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
            'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 
            'MiscFeature', 'SaleType', 'SaleCondition']

print(len(cat_cols))

# Define a Function that Computes and Displays a Frequency Table 
def count_unique(house_prices, cat_cols):
    for col in cat_cols:
        print('\n' + 'For column ' + col + ':')
        print(house_prices[col].value_counts())

count_unique(house_prices, cat_cols)
house_prices.head(2)
house_prices.shape
(house_prices.astype(np.object) == 'NaN').any()
# Counting How many Columns are Null

for col in house_prices.columns:
    if house_prices[col].dtype == object:
        count = 0
        count = [count + 1 for x in house_prices[col] if x == 'NaN']
        print(col + ' ' + str(sum(count)))
house_prices.isnull().any().any()
house_prices.isnull().any()
house_prices.isnull().sum(axis=0)
## Identify and Drop column with too many missing values

house_prices.drop('Alley', axis = 1, inplace = True)
house_prices.drop('FireplaceQu', axis = 1, inplace = True)
house_prices.drop('PoolQC', axis = 1, inplace = True)
house_prices.drop('Fence', axis = 1, inplace = True)
house_prices.drop('MiscFeature', axis = 1, inplace = True)

house_prices.shape
## Remove rows with missing values, accounting for mising values

house_prices = house_prices.dropna()
house_prices.shape
house_prices.isnull().any().any()
house_prices.isnull().sum(axis=0)
# 'LandContour', 'Utilities','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
#  'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
#  'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
#  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
#  'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
#  'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
#  'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'

def plot_bars(house_prices, cols):
    
    # Iterate over the list of columns
    for col in cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(4,4)) # define plot area
        ax = fig.gca() # define axis    
        
        # Compute the counts or frequencies of the categories
        counts = house_prices[col].value_counts() # find the counts for each unique category
        
        # Create a bar plot using the Pandas plot.bar method.
        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Number of Houses by ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

cat_cols = ['LandContour', 'Utilities','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1']

plot_bars(house_prices, cat_cols) 
def plot_bars(house_prices, cols):
    
    # Iterate over the list of columns
    for col in cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(4,4)) # define plot area
        ax = fig.gca() # define axis    
        
        # Compute the counts or frequencies of the categories
        counts = house_prices[col].value_counts() # find the counts for each unique category
        
        # Create a bar plot using the Pandas plot.bar method.
        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Number of Houses by ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

cat_cols = ['Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st']

plot_bars(house_prices, cat_cols)
def plot_bars(house_prices, cols):
    
    # Iterate over the list of columns
    for col in cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(4,4)) # define plot area
        ax = fig.gca() # define axis    
        
        # Compute the counts or frequencies of the categories
        counts = house_prices[col].value_counts() # find the counts for each unique category
        
        # Create a bar plot using the Pandas plot.bar method.
        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Number of Houses by ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

cat_cols = ['Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual']

plot_bars(house_prices, cat_cols)
def plot_bars(house_prices, cols):
    
    # Iterate over the list of columns
    for col in cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(4,4)) # define plot area
        ax = fig.gca() # define axis    
        
        # Compute the counts or frequencies of the categories
        counts = house_prices[col].value_counts() # find the counts for each unique category
        
        # Create a bar plot using the Pandas plot.bar method.
        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Number of Houses by ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

cat_cols = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                    'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional']

plot_bars(house_prices, cat_cols)
def plot_bars(house_prices, cols):
    
    # Iterate over the list of columns
    for col in cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(4,4)) # define plot area
        ax = fig.gca() # define axis    
        
        # Compute the counts or frequencies of the categories
        counts = house_prices[col].value_counts() # find the counts for each unique category
        
        # Create a bar plot using the Pandas plot.bar method.
        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Number of Houses by ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'PavedDrive',  'SaleType', 'SaleCondition']

plot_bars(house_prices, cat_cols)
# 'Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
# 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
# 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
# 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
# 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 
# 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
# 'PoolArea', 'MiscVal','MoSold', 'YrSold'

def plot_histogram(house_prices, num_cols, bins = 10):
    
    # Iterate over the list of columns
    for col in num_cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
       
        # Create a bar plot using the Pandas plot.bar method.
        house_prices[col].plot.hist(ax = ax, bins = bins) # Use the plot.bar method on the counts data frame
        ax.set_title('Histogram of ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

num_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
                    'OverallCond', 'YearBuilt']    
plot_histogram(house_prices, num_cols)
def plot_density_hist(house_prices, num_cols, bins = 10, hist = False):
    
    # Iterate over the list of columns
    for col in num_cols:
        
        # Set a style for the plot grid.
        sns.set_style("whitegrid")
        
        # Define the plot type with distplot using the engine-size column as the argument.
        sns.distplot(house_prices[col], bins = bins, rug=True, hist = hist)
        plt.title('Histogram of ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('Number of Houses')# Set text for y axis
        plt.show()
        
plot_density_hist(house_prices, num_cols)
def plot_histogram(house_prices, num_cols, bins = 10):
    
    # Iterate over the list of columns
    for col in num_cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
       
        # Create a bar plot using the Pandas plot.bar method.
        house_prices[col].plot.hist(ax = ax, bins = bins) # Use the plot.bar method on the counts data frame
        ax.set_title('Histogram of ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

num_cols = ['YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF']    
plot_histogram(house_prices, num_cols)
def plot_density_hist(house_prices, num_cols, bins = 10, hist = False):
    
    # Iterate over the list of columns
    for col in num_cols:
        
        # Set a style for the plot grid.
        sns.set_style("whitegrid")
        
        # Define the plot type with distplot using the engine-size column as the argument.
        sns.distplot(house_prices[col], bins = bins, rug=True, hist = hist)
        plt.title('Histogram of ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('Number of Houses')# Set text for y axis
        plt.show()
        
plot_density_hist(house_prices, num_cols)
def plot_histogram(house_prices, num_cols, bins = 10):
    
    # Iterate over the list of columns
    for col in num_cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(4,4)) # define plot area
        ax = fig.gca() # define axis    
       
        # Create a bar plot using the Pandas plot.bar method.
        house_prices[col].plot.hist(ax = ax, bins = bins) # Use the plot.bar method on the counts data frame
        ax.set_title('Histogram of ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

num_cols = ['LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr']    
plot_histogram(house_prices, num_cols)
def plot_density_hist(house_prices, num_cols, bins = 10, hist = False):
    
    # Iterate over the list of columns
    for col in num_cols:
        
        # Set a style for the plot grid.
        sns.set_style("whitegrid")
        
        # Define the plot type with distplot using the engine-size column as the argument.
        sns.distplot(house_prices[col], bins = bins, rug=True, hist = hist)
        plt.title('Histogram of ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('Number of Houses')# Set text for y axis
        plt.show()
        
plot_density_hist(house_prices, num_cols)
def plot_histogram(house_prices, num_cols, bins = 10):
    
    # Iterate over the list of columns
    for col in num_cols:
        
        # Define fig and axes
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
       
        # Create a bar plot using the Pandas plot.bar method.
        house_prices[col].plot.hist(ax = ax, bins = bins) # Use the plot.bar method on the counts data frame
        ax.set_title('Histogram of ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of Houses')# Set text for y axis
        plt.show()

num_cols = ['TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 
                    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                    'PoolArea', 'MiscVal','MoSold', 'YrSold']    
plot_histogram(house_prices, num_cols)
