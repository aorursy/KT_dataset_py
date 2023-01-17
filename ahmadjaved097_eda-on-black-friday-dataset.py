import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/BlackFriday.csv')

df.head(4)
df.info()
print('Number of rows in dataset: ', df.shape[0])

print('Number of columns in dataset: ', df.shape[1])
df.drop('User_ID',axis = 1).describe()
print('No. of categorical attributes: ', df.select_dtypes(exclude = ['int64','float64']).columns.size)
print('No. of numerical attributes: ', df.select_dtypes(exclude = ['object']).columns.size)
plt.figure(figsize=(10,6))

sns.heatmap(df.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')

plt.title('Null Values present in the dataset',fontsize=14)

plt.show()
df['Product_Category_2'].fillna(0, inplace = True)

df['Product_Category_3'].fillna(0, inplace = True)
sns.set_style('whitegrid')

df.drop('User_ID',axis=1).hist(figsize = (13,10), color = 'darkgreen')

plt.tight_layout()

plt.show()
plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=1)

plt.show()
plt.figure(figsize=(10,6))

sns.set_style('whitegrid')

sns.distplot(df['Purchase'],kde=False,bins = 30,color='green')

plt.show()
sns.boxplot(x='Gender',y='Purchase', data = df, width=0.4)

plt.show()
df.groupby('Gender').agg({'Purchase':['max','min','mean','median']}).round(3)
plt.figure(figsize=(12,6))

sns.boxplot(x = 'Age',y='Purchase', data = df,palette='hls',

            order=['0-17','18-25','26-35','36-45','46-50','51-55','55+'],width=0.5)

plt.show()
df.groupby('Age').agg({'Purchase':['min','max','mean']}).round(3)
plt.figure(figsize=(14,6))

sns.boxplot(x='Occupation',y='Purchase', data = df, width=0.6)

plt.show()
plt.figure(figsize=(14,6))

sns.boxplot(x='City_Category',y='Purchase', data = df,

            width=0.4,palette='hls',order=['A','B','C'])

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='Marital_Status',y='Purchase', data = df,

            width=0.4,palette='GnBu')

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='Stay_In_Current_City_Years',y='Purchase', data = df,

            width=0.4,palette='hls',order=['1','2','3','4+'])

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='Marital_Status',y='Purchase', data = df,

            width=0.4,palette='hls')

plt.show()
df.groupby('Marital_Status').agg({'Purchase':['min','max','mean']}).round(3)