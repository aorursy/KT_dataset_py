import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.impute import SimpleImputer

from scipy import stats
def overview():

    

    data = pd.read_csv("../input/malnutrition-across-the-globe/country-wise-average.csv")

    data1 = pd.read_csv("../input/malnutrition-across-the-globe/malnutrition-estimates.csv")

    # Print the first 5 lines of data

    print("First 5 lines of data \n\n")

    print(data.head())

    print("\n")

    print(data1.head())

    

    # Print data type

    print("\n\n\nDatatype\n")

    print(data.dtypes)

    print("\n")

    print(data1.dtypes)

    

    # Print number of null values 

    print("\n\n\nNumber of null values\n")

    print(data.isnull().sum())

    print("\n")

    print(data1.isnull().sum())

    

    # Print data summary

    print("\n\n\nData summary\n")

    print(data.describe())

    print("\n")

    print(data1.describe())

    

    # Print data shape

    print("\n\n\nData shape\n")

    print("Data has {} rows and {} columns".format(data.shape[0], data.shape[1]))

    print("\n")

    print("Data1 has {} rows and {} columns".format(data1.shape[0], data1.shape[1]))

    

    return data, data1



data, data1 = overview()
data = data.dropna(subset = ['Wasting', 'Overweight', 'Stunting', 'Underweight'])

data1 = data1.dropna(subset = ['Stunting', 'Underweight', 'Survey Sample (N)', 'Notes'])



imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')

data[['Severe Wasting']] = imputer.fit_transform(data[['Severe Wasting']])

data1[['Severe Wasting', 'Wasting', 'Overweight']] = imputer.fit_transform(data1[['Severe Wasting', 'Wasting', 'Overweight']])
# Check for NaN

print(data.isnull().sum())

print(data1.isnull().sum())
# Create a function to separate out numerical and categorical data

    ## Using this function to ensure that all non-numerical in a numerical column

    ## and non-categorical in a categorical column is annotated

def cat_variable(df):

    return list(df.select_dtypes(include = ['category', 'object']))



def num_variable(df):

    return list(df.select_dtypes(exclude = ['category', 'object']))



categorical_variable = cat_variable(data)

numerical_variable = num_variable(data)

categorical_variable1 = cat_variable(data1)

numerical_variable1 = num_variable(data1)



# Create a function to process outlier data

def outlier(data):

    z = np.abs(stats.zscore(data[numerical_variable]))

    z_data = data[(z < 3).all(axis=1)] # Remove any outliers with Z-score > 3 or < -3

    return z_data



data = outlier(data)

data1 = outlier(data1)
# Replace float to int

data['Income Classification'] = data['Income Classification'].astype('int')



# Create a new column to represent income level

def func(row):

    if row == 0:

        return 'Low income'

    elif row == 1:

        return 'Lower middle income'

    elif row == 2:

        return 'Upper middle income'

    else:

        return 'High income'



data['Income level'] = data.apply(lambda x: func(x['Income Classification']), axis=1)



# Plot countplot

plt.figure(figsize = (10,8))

sns.countplot(data = data, x = 'Income level').set_title('Income level')
plt.figure(figsize = (10,8))

sns.boxplot(data = data, x = 'Income level', y = 'Severe Wasting').set_title("Severe wasting among different income")
plt.figure(figsize = (10,8))

sns.boxplot(data = data, x = 'Income level', y = 'Wasting').set_title("Wasting among different income")
plt.figure(figsize = (10,8))

sns.boxplot(data = data, x = 'Income level', y = 'Overweight').set_title("Overweight among different income")
plt.figure(figsize = (10,8))

sns.boxplot(data = data, x = 'Income level', y = 'Stunting').set_title("Stunting among different income")
plt.figure(figsize = (10,8))

sns.boxplot(data = data, x = 'Income level', y = 'Underweight').set_title("Underweight among different income")
# Choosing the country

bangladesh = data1[data1['Country'] == 'BANGLADESH']



# Indexing the year

bangladesh = bangladesh.set_index('Year')

# Creating time series

axes = bangladesh[["Severe Wasting", "Wasting", "Overweight", "Stunting", "Underweight"]].plot(figsize=(11, 9), subplots=True, linewidth=1)
# Choosing the country

kuwait = data1[data1['Country'] == 'KUWAIT']



# Indexing the year

kuwait = kuwait.set_index('Year')

# Creating time series

axes = kuwait[["Severe Wasting", "Wasting", "Overweight", "Stunting", "Underweight"]].plot(figsize=(11, 9), subplots=True, linewidth=1)
# Choosing the country

chile = data1[data1['Country'] == 'CHILE']



# Indexing the year

chile = chile.set_index('Year')

# Creating time series

axes = chile[["Severe Wasting", "Wasting", "Overweight", "Stunting", "Underweight"]].plot(figsize=(11, 9), subplots=True, linewidth=1)