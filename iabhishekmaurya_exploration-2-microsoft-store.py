# Importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# Reading the dataset

data = pd.read_csv('../input/windows-store/msft.csv')

data.head(10)
print("Shape of data: ",data.shape)
data.info()
data.describe()
data['Price'].value_counts()
# Looking ate the unique values of Categorical Features

print(data['Rating'].unique())

print(data['Category'].unique())
# Count of Null Data

data.isnull().sum()
print("Shape of data Before dropping any Row: ",data.shape)

data = data[data['Rating'].notna()]

print("Shape of data After dropping Rows with NULL values in Rating: ",data.shape)

data = data[data['Category'].notna()]

print("Shape of data After dropping Rows with NULL values in Category : ",data.shape)

data = data[data['Date'].notna()]

print("Shape of data After dropping Rows with NULL values in Date  : ",data.shape)

data = data[data['Price'].notna()]

print("Shape of data After dropping Rows with NULL values in Price  : ",data.shape)
data['Date'].head()
for i in range(data.shape[0]):

    data.at[i, 'Year'] = data['Date'][i].split('-')[-1]

    data.at[i, 'Month'] = data['Date'][i].split('-')[1]
data.head()
data['Year'] = data['Year'].astype(int)

data['Month'] = data['Month'].astype(int)
data['Price'].head()
for i in range(data.shape[0]):

    if data['Price'][i] != 'Free':

        data.at[i, 'Price (in ₹)'] = data['Price'][i].split()[-1]

        data.at[i, 'Price (in ₹)'] = data.at[i, 'Price (in ₹)'].replace(",","")
data['Price (in ₹)'] = data['Price (in ₹)'].astype(float)
for i in range(data.shape[0]):

    if data['Price'][i] != 'Free':

        data.at[i, 'Price'] = 'Paid'
data = data.rename(columns={"Price":"Sell Type"})

data.head()
data['Sell Type'].value_counts()
data.info()
data['Rating'].describe()
rat_d = data.groupby(['Rating'])[['No of people Rated']].sum()

rat_d.reset_index(level=0, inplace=True)



fig, ax1 = plt.subplots(figsize = (8,8))

ax1.pie(rat_d['No of people Rated'], labels = rat_d['Rating'], autopct='%1.1f%%', 

        shadow=True, startangle=90)

ax1.set_title('Rating Distribution', fontsize=15)

ax1.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout() 
data['Year'].describe()
var = 'Year'

y_r_data = pd.concat([data['Rating'], data[var]], axis=1)

f, ax = plt.subplots(figsize=(15, 8))

fig = sns.boxplot(x=var, y="Rating", data=data)

fig.axis(ymin=0, ymax=7);

plt.xticks(rotation=90);
var = 'Year'

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (20,10))



sns.countplot(x = var,palette="Blues", data = data, ax=ax1)

sns.distplot(data[var], ax=ax2)

sns.boxplot(x = var, data = data, orient = 'v', ax = ax3)

sns.countplot(x = var,hue="Rating", data = data, ax=ax4)



ax1.set_xlabel(var, fontsize=15)

ax1.set_ylabel('Count', fontsize=15)

ax1.set_title('Year Sales Count Distribution', fontsize=15)

ax1.tick_params(labelsize=15)



ax2.set_xlabel('Year', fontsize=15)

ax2.set_ylabel('Occurence', fontsize=15)

ax2.set_title('Year x Ocucurence', fontsize=15)

ax2.tick_params(labelsize=15)



ax3.set_xlabel('Year Sales', fontsize=15)

ax3.set_ylabel(var, fontsize=15)

ax3.set_title('Year Sales Count Distribution', fontsize=15)

ax3.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout() 
data['Month'].value_counts()
plt.figure(figsize=(16, 10))

sns.set(style="whitegrid")

columns = ["Rating"]

for col in columns:

    x = data.groupby("Year")[col].mean()

    ax= sns.lineplot(x=x.index,y=x,label=col)

ax.set_title('Rating over year')

ax.set_ylabel('Rating')

ax.set_xlabel('Year')
sell_t = data.groupby('Sell Type')['No of people Rated'].sum()

sell_t.reset_index()

sell_t.plot.pie(subplots=True, figsize=(8, 8))



plt.subplots_adjust(wspace=0.5)

plt.tight_layout() 
cate_d = data.groupby(['Category'])[['No of people Rated']].sum()

cate_d.reset_index(level=0, inplace=True)



fig, ax1 = plt.subplots(figsize = (8,8))

ax1.pie(cate_d['No of people Rated'], labels = cate_d['Category'], autopct='%1.1f%%', 

        shadow=True, startangle=90)

ax1.set_title('Category Distribution', fontsize=15)

ax1.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout() 