import numpy as np        # Fundamental package for linear algebra and multidimensional arrays

import pandas as pd       # Data analysis and manipultion tool
# In read_csv() function, we have passed the location to where the files are located in the UCI website. The data is separated by ';'

# so we used separator as ';' (sep = ";")

red_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

white_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
# Red Wine

red_wine_data.head()    # we can also pass the number of records we want in the brackets (). By default it displays first 5 records.
# White Wine

white_wine_data.head(6)      # we will get first 6 records from white wine data
# Columns / Attribute

red_wine_data.columns
# Add a column to separate whether the wine is red or white.

red_wine_data['color'] = 'r'

white_wine_data['color'] = 'w'

wine_data = pd.concat([red_wine_data, white_wine_data])
# rename() function is used to rename the columns



# wine data

# red_wine_data

wine_data.rename(columns={'fixed acidity': 'fixed_acidity', 'citric acid':'citric_acid', 'volatile acidity':'volatile_acidity',

                          'residual sugar':'residual_sugar', 'free sulfur dioxide':'free_sulfur_dioxide', 'total sulfur dioxide':'total_sulfur_dioxide'},

                 inplace = True)



# red_wine_data

red_wine_data.rename(columns={'fixed acidity': 'fixed_acidity', 'citric acid':'citric_acid', 'volatile acidity':'volatile_acidity',

                          'residual sugar':'residual_sugar', 'free sulfur dioxide':'free_sulfur_dioxide', 'total sulfur dioxide':'total_sulfur_dioxide'},

                 inplace = True)    # inplace = True makes changes in the dataframe itself



# white_wine_data

white_wine_data.rename(columns={'fixed acidity': 'fixed_acidity', 'citric acid':'citric_acid', 'volatile acidity':'volatile_acidity',

                          'residual sugar':'residual_sugar', 'free sulfur dioxide':'free_sulfur_dioxide', 'total sulfur dioxide':'total_sulfur_dioxide'},

                 inplace = True)
red_wine_data.head(2)
# concise summary about dataset

red_wine_data.info()
white_wine_data.info()
# Basic Statistical details 

red_wine_data.describe()
white_wine_data.describe()
# first import data visualizations libraries

import matplotlib.pyplot as plt

import seaborn as sns



# To ignore warnings

import warnings

warnings.filterwarnings('ignore')
# red wines      

red_wine_data.hist(bins=10, figsize=(16,12))

plt.show()
# white wine

white_wine_data.hist(bins=10, figsize=(16,12))

plt.show()
# Creating pivot table for red wine

columns = list(red_wine_data.columns).remove('quality')

red_wine_data.pivot_table(columns, ['quality'], aggfunc=np.median)    # By default the aggfunc is mean
# Creating pivot table for red wine

columns = list(white_wine_data.columns).remove('quality')

white_wine_data.pivot_table(columns, ['quality'], aggfunc=np.median)
# red wines

red_wine_data.corr()
# white wines

white_wine_data.corr()
# red wines

plt.figure(figsize=(16, 12))

sns.heatmap(red_wine_data.corr(), cmap='bwr', annot=True)     # annot = True: to display the correlation value in the graph
# white wines

plt.figure(figsize=(16, 12))

sns.heatmap(white_wine_data.corr(), cmap='bwr', annot=True)
# Countplot for quality of wines present in different category of wines (red and white)

plt.figure(figsize=(12,8))

sns.countplot(wine_data.quality, hue=wine_data.color)
# red wine

sns.pairplot(red_wine_data)
# Pairplot

sns.pairplot(white_wine_data)