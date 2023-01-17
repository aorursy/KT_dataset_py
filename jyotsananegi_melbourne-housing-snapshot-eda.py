# importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime # datetime processing



# Data visualisation

import matplotlib.pyplot as plt

import seaborn as sns



# setting path of the dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

melb_house = pd.read_csv("/kaggle/input/melbourne-housing-snapshot/melb_data.csv")



# checking the columns

melb_house.head()
melb_house.shape
melb_house.info()
# checking for unique entries



unique_val = []

for i in melb_house.columns:

    u = melb_house[i].nunique()

    unique_val.append(u)

    

pd.DataFrame({"No. of unique values": unique_val}, index=melb_house.columns)
# Working dataset

dataset = melb_house.copy()
# plot of missing value

plt.figure(figsize=(9,5))

sns.heatmap(dataset.isnull(),yticklabels=False, cbar=False, cmap="Paired");

plt.title("Heatmap of Missing Values");
# Features with missing values

miss = dataset.isnull().sum().sort_values(ascending = False).head(5)

miss_per = (miss/len(dataset))*100



# Percentage of missing values

pd.DataFrame({'No. missing values': miss, '% of missind data': miss_per.values})
dataset['Car'].value_counts()
# Filling null value

dataset['Car'].fillna(0, inplace = True)



# confimation after filling the null values

print("Null values before replacement :", melb_house['Car'].isnull().sum())

print("Null values after replacement :", dataset['Car'].isnull().sum())
dataset['CouncilArea'].value_counts()
# Filling the null value 

dataset['CouncilArea'].fillna('Unavailable', inplace = True)





# confimation after filling the null values

print("Null values before replacement :", melb_house['CouncilArea'].isnull().sum())

print("Null values after replacement :", dataset['CouncilArea'].isnull().sum())
# Filling the null value 

dataset['YearBuilt'].fillna("Unknown", inplace=True)



# confimation after filling the null values

print("Null values before replacement :", melb_house['YearBuilt'].isnull().sum())

print("Null values after replacement :", dataset['YearBuilt'].isnull().sum())
plt.figure(figsize = (5, 5))

sns.distplot(dataset['BuildingArea']);
# Filling the null value 

dataset['BuildingArea'].fillna(0, inplace = True)



# confimation after filling the null values

print("Null values before replacement :", melb_house['BuildingArea'].isnull().sum())

print("Null values after replacement :", dataset['BuildingArea'].isnull().sum())
# log transformation of price

dataset['Price_trans'] = np.log(dataset['Price'])



# plot of price

plt.figure(figsize=(10, 5))

plt.subplots_adjust(wspace=0.5)

plt.suptitle("Distribution of Price", fontsize=14)



plt.subplot(1,2,1)

p1 = sns.kdeplot(dataset['Price']);

p1.title.set_text("Before Transfromation")



plt.subplot(1,2,2)

p2 = sns.kdeplot(dataset['Price_trans']);

p2.title.set_text("After log Transformation")
# Grouping the numerical data

num =  dataset.select_dtypes(exclude="object")

num = num.drop(['Price'], axis=1)



# Distributions of numrical data

plt.figure(figsize=(15, 13))

plt.subplots_adjust(hspace=0.4, wspace=0.3)



j=1



for i in list(num.columns):

    plt.subplot(4,3,j)

    sns.distplot(dataset[i])

    j+=1

    

plt.suptitle("Distribution of Numerical Data", fontsize=15);



plt.figure(figsize=(15, 5))

plt.subplots_adjust(wspace=0.5)



plt.subplot(1,2,1)

ax1 = sns.countplot(dataset['Type'], palette='Accent');

ax1.title.set_text("Plot of House Type")



plt.subplot(1,2,2)

ax2= sns.countplot(dataset['Method'], palette='Accent');

ax2.title.set_text("Plot of Method of Selling")

    
plt.figure(figsize=(10, 5))

sns.countplot(y = dataset['Regionname']);

plt.ylabel("Region Name", fontsize=12);

plt.xlabel("Count", fontsize=12);
# checking for top 10 seller

dataset['SellerG'].value_counts().head(10).plot(kind='bar', color='brown');



# plot for top seller

plt.title("Top 10 Estate Agents", fontsize=14);
# coverting date into datetime format

dataset['Date'] = pd.to_datetime(dataset['Date'])

year = dataset['Date'].map(lambda x: datetime.strftime(x, '%Y'))

dataset['year'] = year

month = dataset['Date'].map(lambda x: datetime.strftime(x, '%b'))

dataset['month'] = month



# plot of each month

plt.figure(figsize = (12, 4))

sns.countplot(dataset['month'], hue=dataset['year'], palette='Set1');
plt.figure(figsize=(10, 5))

sns.boxplot(x = 'Rooms', y = "Price_trans", data=dataset);

plt.ylabel("Price");
plt.figure(figsize=(15, 5))

plt.subplots_adjust(wspace=0.3)



plt.subplot(1,2,1)

sns.boxplot(x = 'Type', y = "Price_trans", data=dataset, palette='Set3');

plt.ylabel("Price");



plt.subplot(1,2,2)

sns.boxplot(x = 'Method', y = "Price_trans", data=dataset, palette='Set3');

plt.ylabel("Price");
plt.figure(figsize=(15, 5))

sns.boxenplot(x = 'Car', y = "Price_trans", data=dataset);

plt.ylabel("Price");
sns.lmplot(x="Landsize", y='Price_trans', data=dataset);

plt.ylabel("Price");

plt.xticks(rotation=15);
sns.scatterplot(x='Distance', y='Price_trans', data=dataset);

plt.ylabel("Price");
# Dataset of Metropolitan area

rm = dataset[dataset['Regionname'].map(lambda x: 'Metropolitan' in x)]



# Dataset of Victoria area

rv = dataset[dataset['Regionname'].map(lambda x: 'Victoria' in x)]



# plots of both region

plt.figure(figsize=(15, 5))



plt.subplot(1,2,1)

ar1 = sns.scatterplot(x='Distance', y='Price_trans', data=rm, color='orange');

plt.ylabel("Price");

ar1.title.set_text("Metropolitan");



plt.subplot(1,2,2)

ar2 = sns.scatterplot(x='Distance', y='Price_trans', data=rv, color="green");

plt.ylabel("Price");

ar2.title.set_text('Victoria');