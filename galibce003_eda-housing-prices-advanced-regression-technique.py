import pandas as pd               #Data Manipulation
import numpy as np                #Linear Algebra
import matplotlib.pyplot as plt   #Visualization
import seaborn as sns             #Visualization
# Load the dataset
df = pd.read_csv('../input/housing-prices-advanced-regression-techniques/train.csv')

# When a dataset has a large amount features, the features located in the middle can't be seen in notebook.
# They are indicated by some dots.
# But if all the features need to be seen, this one line of code will be there to help you out. 
pd.pandas.set_option('display.max_columns', None)
# Head : 1st few rows of the dataset
df.head()
# Shape of the dataset
df.shape
# Data types
df.dtypes
df.isnull().sum().sort_values(ascending = False)[0:25]
plt.figure(figsize= (20, 5))
sns.heatmap(df.isnull(), yticklabels = False, cmap = 'viridis')
plt.show()
df[df.columns[1:]].corr()['SalePrice'][:]
# We have craeted a list with all the features which data types aren't 'Object'
numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']

# Number of numerical features
print('Number of numerical features : {}'.format(len(numerical_features)))

# List of numerical features
numerical_features
df[numerical_features].head()
year_features = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature or 'yr' in feature or 'year' in feature]

# Number of year features
print('Number of year features : {}\n'.format(len(year_features)))

# List of year features
year_features
for i in year_features: 
    df.groupby(i)['SalePrice'].median().plot()
    plt.show()
# For choosing the descrete features, we set the threshold value to 30
# What does it mean? We have choosen only those features which have the number of unique values less than 30.
# And also which are not in the year_features.
descrete_features = [i for i in numerical_features if len(df[i].unique()) < 30 and i not in year_features]

print('Number of descrete features : {}\n'.format(len(descrete_features)))

descrete_features
for i in descrete_features:
    df.groupby(i)['SalePrice'].median().plot.bar()
    plt
    plt.show()
# Numerical_features which aren't in descrete_features, year_features and Id column, are listed as Continuous Features
continuous_features = [i for i in numerical_features if i not in descrete_features+year_features+['Id']]

print('Number of continuous features : {}\n'.format(len(continuous_features)))

continuous_features
for i in continuous_features:
    df[i].hist(bins= 25)
    plt.xlabel(i)
    plt.show()
for feature in continuous_features:
    if 0 in df[feature].unique():          # We will skip the features which have any unique values of 0
        pass                               # Because the value of log0 is infinite
    else:
        df[feature] = np.log(df[feature])  # Converted into logarithmic scale using numpy
        df[feature].hist(bins = 25)
        plt.xlabel(feature)
        plt.show()
for feature in continuous_features:
    if 0 in df[feature].unique():
        pass
    else:
        df[feature] = np.log(df[feature])
        plt.scatter(df[feature], df['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('Sales Price')
        plt.show()
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

print('\nNumber of categorical features : {}'.format(len(categorical_features)))

categorical_features
df[categorical_features].head()
for i in categorical_features:
    print('Feature : {}\nUnique values : {}\nNumber of unique values : {}\n'.format(i, df[i].unique(), len(df[i].unique())))
for i in categorical_features:
    df.groupby(i)['SalePrice'].median().plot.bar()
    plt.show()
for feature in continuous_features:
    if 0 in df[feature].unique():
        pass
    else:
        df[feature] = np.log(df[feature])
        df.boxplot(column = feature)
        plt.show()