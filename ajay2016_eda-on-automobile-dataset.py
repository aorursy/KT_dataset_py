# Loading necessary packages
import numpy as np
import pandas as pd
auto = pd.read_csv('../input/Automobile_data.csv')
# Display top 5 rows
auto.head(5)
auto.shape  # shape gives the dimensionality of the data frame .so here we have 205 rows 26 columns
auto.isnull().sum() # lets see for any null values in the dataset.
auto.dtypes
auto[auto['normalized-losses']=='?'].count() # Normalized-losses has 41 '?'s
# converting the ? with the mean values in the 'normalize-losses' column
nl=auto['normalized-losses'].loc[auto['normalized-losses'] !='?'].count()
nmean=nl.astype(int).mean()
auto['normalized-losses']=auto['normalized-losses'].replace('?',nmean).astype(int)
auto.head(7)
# Prices column
# checking how many rows are numeric and how many aren't.
auto['price'].str.isnumeric().value_counts()

# Check out the values which are not numeric
auto['price'].loc[auto['price'].str.isnumeric() == False]

# Setting the missing value to mean of price and convert the datatype to integer
price = auto['price'].loc[auto['price'] != '?']
pmean = price.astype(str).astype(int).mean()
auto['price'] = auto['price'].replace('?',pmean).astype(int)
auto['price'].head()
# Cleaning Horsepower
# checking how many rows are numeric and how many aren't.
auto['horsepower'].str.isnumeric().value_counts()
# Check out the values which are not numeric
auto['horsepower'].loc[auto['horsepower'].str.isnumeric() == False]
# replacing the horsepower misssing values

horsepower = auto['horsepower'].loc[auto['horsepower'] != '?']
hpmean = horsepower.astype(str).astype(int).mean()
auto['horsepower'] = auto['horsepower'].replace('?',pmean).astype(int)
# cleaning the bore

# Find out the number of invalid value
auto['bore'].loc[auto['bore'] == '?']

# Replace the non-numeric value to null and conver the datatype
auto['bore'] = pd.to_numeric(auto['bore'],errors='coerce') # converting from Object to numeric and use of 'coerce' -> invalid parsing will be set as NaN
#cleaning stoke the similar way
auto['stroke'] = pd.to_numeric(auto['stroke'],errors='coerce')
auto.dtypes
#Cleaning the peak rpm data

# Convert the non-numeric data to null and convert the datatype
auto['peak-rpm'] = pd.to_numeric(auto['peak-rpm'],errors='coerce')
auto.dtypes
# cleaning the num-of-doors data
# remove the records which are having the value '?'

auto['num-of-doors'].loc[auto['num-of-doors']=='?']
auto=auto[auto['num-of-doors']!='?']
auto['num-of-doors'].loc[auto['num-of-doors'] == '?'] # will give no values as both the entries has been deleted.
auto.isnull().sum() # Now we have 3 columns with NaN's.
pd.set_option('display.max_columns', None) # This will display all the columns.
auto.head(4)
auto.describe() 
#To do plottting we need to import Matplot library.

import matplotlib.pyplot as plt
% matplotlib inline    
#here inline means plotting within the notebook

auto.make.value_counts().nlargest(23).plot(kind='bar',figsize=(24,6)) # here there are 23 unique values under column -'make'. ]
# So we display all 23 of them. nomrally 10 will be a good number for visualization
plt.title("No. of Vehicles in terms of Make")
plt.xlabel("Make")
plt.ylabel("No. of Vehicles")
# Insurance risk ratings Histogram. 

auto.symboling.hist(bins=6)
plt.title("Insurance risk ratings of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Risk rating');
#Normalized losses histogram

auto['normalized-losses'].hist(bins=6,color='Green',grid=False) # grid removes the lines
plt.title("Normalized losses of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Normalized losses')
#Fuel Type - Bar Graph

auto['fuel-type'].value_counts().plot(kind='bar',color='grey')
plt.title("Fuel type frequence diagram")
plt.ylabel('No. of Vehicles')
plt.xlabel('Fule type');
# Fuel type pie diagram
auto['aspiration'].value_counts().plot.pie(figsize=(6, 6), autopct='%.2f')
plt.title("Fuel type pie diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Fuel type');
auto.horsepower[np.abs(auto.horsepower-auto.horsepower.mean())<=(3*auto.horsepower.std())].hist(bins=5,color='red');
plt.title("Horse power histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Horse power')
#Curb weight histogram

auto['curb-weight'].hist(bins=6,color='blue') 
plt.title("curb weight histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Curb weight');
# Drive wheels bar chart

auto['drive-wheels'].value_counts().plot(kind='bar',color='purple')
plt.title("Drive wheels diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Drive wheels');
#Number of doors bar chart

auto['num-of-doors'].value_counts().plot(kind='bar',color='black')
plt.title("Number of doors frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Number of doors');
# no of cylinders Bar Graph

auto['num-of-cylinders'].value_counts().plot(kind='bar',color='green')
plt.title("Number of cylinders frequency diagram")
plt.ylabel('Number of Vehicles')
plt.xlabel('Number of Cylinders');
#Body Style Bar Graph

auto['body-style'].value_counts().plot(kind='bar',color='brown')
plt.title("Body Style frequency diagram")
plt.ylabel('Number of Vehicles')
plt.xlabel('Body Style');
import seaborn as sns
corr=auto.corr()
sns.set_context("notebook",font_scale=1.0,rc={"lines.linewidth":3})
plt.figure(figsize=(13,7))
a=sns.heatmap(corr,annot=True,fmt ='.2f')
# Create a Boxplot of Price and make

plt.rcParams['figure.figsize']=(22,9)
sns.boxplot(data=auto,x='make',y='price');
# Scatter plot of price and engine size:

sns.lmplot('price',"engine-size",data=auto);
#Scatter plot of normalized losses and symboling

sns.lmplot('normalized-losses',"symboling",data=auto)
# Scatter plot of engine size and city MPH

plt.scatter(auto['engine-size'],auto['city-mpg'])
plt.xlabel('Engine size')
plt.ylabel('City MPG');
g = sns.lmplot('city-mpg',"curb-weight", auto, hue="make", fit_reg=False);
h = sns.lmplot('highway-mpg',"curb-weight", auto, hue="make", fit_reg=False);
# Drive wheels and City MPG bar chart

auto.groupby('drive-wheels')['city-mpg'].mean().plot(kind='bar',color='blue')
plt.title("Drive wheels City MPG")
plt.ylabel('City MPG')
plt.xlabel('Drive wheels');
plt.rcParams['figure.figsize']=(10,7)
sns.boxplot(x='drive-wheels',y='price',data=auto);
# Normalized losses based on no. of doors and body style

import pandas as pd
pd.pivot_table(auto, index=['num-of-doors','body-style'],values='normalized-losses').plot(kind='bar',color='green')
plt.title("Normalized losses based on body style and no. of doors")
plt.ylabel('Normalized losses')
plt.xlabel('Body style and No. of doors');