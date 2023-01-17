# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/BlackFriday.csv')
df.head()
#Show datatypes of features
df.info()
#Key summary statistics of features
df.describe()
#Shows which features have missing data
import missingno as msno
msno.bar(df)
#Count of null values
print(df.isnull().sum())
#Listing unique values in Product_Category_2
list(df['Product_Category_2'].unique())
#Replace NaN values with '0' for Product_Category_2
df['Product_Category_2'].fillna('0', inplace=True)
list(df['Product_Category_2'].unique())
#Dropping Product_Category_3 because 1/3 of data is missing
df1 = df.drop(columns=['Product_Category_3'])
df1.columns
#Changing features to proper data types
df1['User_ID'] = df1['User_ID'].astype('object')
df1['Occupation'] = df1['Occupation'].astype('int').astype('str').astype('category')
df1['Marital_Status'] = df1['Marital_Status'].astype('category')
df1['City_Category'] = df1['City_Category'].astype('category')
df1['Age'] = df1['Age'].astype('category')
df1['Product_Category_1'] = df1['Product_Category_1'].astype('int').astype('str').astype('category')
df1['Product_Category_2'] = df1['Product_Category_2'].astype('int').astype('str').astype('category')
df1.info()
df1.head()
#Number of unique values for each feature
df1.nunique()
#Listing unique values for each feature
for col in df1:
    print(col + ":")
    print(df1[col].unique())
#Heatmap correlation of continuous features
plt.figure(figsize=(10,10))
sns.heatmap(df1.corr(),annot=True,fmt =".2f") #annot=True -> means write data value on each cell
plt.title('Correlation between Features')
#Checking for outliers
plt.figure(figsize=(5,5))
sns.boxplot(y=df1['Purchase'])
plt.yticks(np.arange(0, 25000, 2500))
plt.title('Distribution of Purchase')
#Number of outliers
df1[df1['Purchase'] > 20000].count()
#Removal of outliers
df1 = df1[df1['Purchase'] <= 20000]
df1.info()
plt.figure(figsize=(10,5))
sns.boxplot(y=df1['Purchase'], x=df1['Age'])
plt.title('Distribution of Purchase by Age')
plt.figure(figsize=(5,5))
sns.countplot(df1['Age'], hue=df1['Gender'])
plt.title('Distribution of Age by Gender')
#People aged between 26-35 are making the most purchases across all cities
#The elderly from city C seems to be making more purchases
sns.catplot(x='Age', y='Purchase', hue='Gender', col='City_Category', data=df1, 
            kind='bar', height=3, aspect=1.2, order=['0-17','18-25','26-35','36-45','46-50','51-55','55+'],
            col_order=['A','B','C'], estimator=sum)
grouped_gender_marital_status = df1.groupby(['Age','Gender','Marital_Status'])['User_ID'].count().unstack()
print(grouped_gender_marital_status)
#Young singles are making more purchases compared to married couples but there is a reversal of trend
#from age 46 onwards
sns.catplot(x='Age', y='Purchase', hue='Gender', col='City_Category', row='Marital_Status', data=df1, 
            kind='bar', height=3, aspect=1.2, order=['0-17','18-25','26-35','36-45','46-50','51-55','55+'],
            col_order=['A','B','C'], estimator=sum)
pivot_occupation = pd.pivot_table(df1, index='Occupation', columns='Age', values='Purchase', aggfunc=np.sum)
plt.figure(figsize=(20,10))
sns.heatmap(pivot_occupation, cmap='Blues', robust=True)
#Product 1 appears to be a very popular item in general
#Products 5 and 8 seem to be more popular with younger people
pivot_category = pd.pivot_table(df1, index='Product_Category_1', columns='Age', values='Purchase', aggfunc=np.sum)
plt.figure(figsize=(20,10))
sns.heatmap(pivot_category, cmap='Blues', robust=True)