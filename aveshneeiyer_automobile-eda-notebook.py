# DS L2T16

# Automobile EDA (and Automobile Report in task 16 folder)
# Import packages for calcs & visualisations



import numpy as np

import pandas as pd

import re



# For Box-Cox Transformation

from scipy import stats



# for min_max scaling

from mlxtend.preprocessing import minmax_scaling



# Data visualization

import seaborn as sns

import missingno

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# Load Automobile dataset and display the 1st 5 rows

df = pd.read_csv('../input/automobile/automobile.txt')

df.head()
# Display summarised data re dataframe

df.info()
# View Features

df.columns
# Stats of the data set

df.describe()
# Visualise missing data

missingno.matrix(df, figsize = (13,3))
# Number of missing data points per column

missing_values_count = df.isnull().sum()



# The no of missing points in the columns

missing_values_count[0:27]
# List of columns that contain a "?" for missing data

cols = list(df.columns)

for col in cols:

    if('?' in df[col].value_counts()):

        print(col + " - " + str(df[col].value_counts()['?']))
# Cleaning the NORMALISED LOSSES field

# Find out number of records having 'NaN ""' value for normalized losses

df['normalized-losses'].loc[df['normalized-losses'] == '?'].count()
# Setting the missing value to mean of normalized losses and convert the datatype to integer

nl = df['normalized-losses'].loc[df['normalized-losses'] != '?']

nlmean = nl.astype(str).astype(int).mean()

df['normalized-losses'] = df['normalized-losses'].replace('?',nlmean).astype(int)

df['normalized-losses'].head()
# Cleaning the PRICE data

# Find out the number of values which are not numeric

df['price'].str.isnumeric().value_counts()
# List out the values which are not numeric

df['price'].loc[df['price'].str.isnumeric() == False]
#Setting the missing value to mean of price and convert the datatype to integer

price = df['price'].loc[df['price'] != '?']

pmean = price.astype(str).astype(int).mean()

df['price'] = df['price'].replace('?',pmean).astype(int)

df['price'].head()
# Cleaning the HORSEPOWER

# Checking the numeric and replacing with mean value and convert the datatype to integer

df['horsepower'].str.isnumeric().value_counts()

horsepower = df['horsepower'].loc[df['horsepower'] != '?']

hpmean = horsepower.astype(str).astype(int).mean()

df['horsepower'] = df['horsepower'].replace('?',pmean).astype(int)
#Checking the outlier of horsepower

df.loc[df['horsepower'] > 10000]
#Excluding the Outlier data for horsepower

df[np.abs(df.horsepower-df.horsepower.mean())<=(3*df.horsepower.std())]
# Cleaning BORE

# Find out the number of invalid values

df['bore'].loc[df['bore'] == '?']
# Replace the non-numeric value to null and convert the datatype

df['bore'] = pd.to_numeric(df['bore'],errors='coerce')

df.dtypes
# Cleaning the STROKE

# Replace the non-number value to null and convert the datatype

df['stroke'] = pd.to_numeric(df['stroke'],errors='coerce')

df.dtypes
# Cleaning the PEAK RPM

# Convert the non-numeric data to null and convert the datatype

df['peak-rpm'] = pd.to_numeric(df['peak-rpm'],errors='coerce')

df.dtypes
# Cleaning the num-of-doors data

# remove the records which are having the value '?'

df['num-of-doors'].loc[df['num-of-doors'] == '?']

df = df[df['num-of-doors'] != '?']

df['num-of-doors'].loc[df['num-of-doors'] == '?']
# Vehicle make frequency

df.make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))

plt.title("Number of vehicles by make")

plt.ylabel('Number of vehicles')

plt.xlabel('Make');
# Insurance risk ratings histogram

df.symboling.hist(bins=6,color='green');

plt.title("Insurance risk ratings of vehicles")

plt.ylabel('Number of vehicles')

plt.xlabel('Risk rating');
# Normalised losses Histogram

df['normalized-losses'].hist(bins=5,color='orange');

plt.title("Normalized losses of vehicles")

plt.ylabel('Number of vehicles')

plt.xlabel('Normalized losses');
# Fuel type

df['fuel-type'].value_counts().plot(kind='bar',color='purple')

plt.title("Fuel type frequency diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('Fuel type');
df['aspiration'].value_counts().plot.pie(figsize=(6, 6), autopct='%.2f')

plt.title("Fuel type pie diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('Fuel type');
# Heatmap showing correlation between features

import seaborn as sns

corr = df.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

a = sns.heatmap(corr, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
# Price and Make Box Plot

plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="make", y="price", data=df)
# Scatter plot of price and engine size

g = sns.lmplot('price',"engine-size", df);
# Scatter plot of normalized losses and symboling

g = sns.lmplot('normalized-losses',"symboling", df);
# Drive wheels and City MPG bar chart

df.groupby('drive-wheels')['city-mpg'].mean().plot(kind='bar', color = 'peru');

plt.title("Drive wheels City MPG")

plt.ylabel('City MPG')

plt.xlabel('Drive wheels');
# Drive wheels and Highway MPG bar chart

df.groupby('drive-wheels')['highway-mpg'].mean().plot(kind='bar', color = 'peru');

plt.title("Drive wheels Highway MPG")

plt.ylabel('Highway MPG')

plt.xlabel('Drive wheels');
# Boxplot of Drive wheels and Price

plt.rcParams['figure.figsize']=(10,5)

ax = sns.boxplot(x="drive-wheels", y="price", data=df)
# Normalized losses based on body style and no. of doors

pd.pivot_table(df,index=['body-style','num-of-doors'], values='normalized-losses').plot(kind='bar',color='purple')

plt.title("Normalized losses based on body style and no. of doors")

plt.ylabel('Normalized losses')

plt.xlabel('Body style and No. of doors');