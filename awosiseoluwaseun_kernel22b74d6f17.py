#importing libraries

import numpy as np

import pandas as pd





import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

%matplotlib inline

plt.style.use('ggplot')

pylab.rcParams['figure.figsize'] = 12,8



import seaborn as sns

sns.set(style='darkgrid')





from sklearn import linear_model

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')





#Read the dataset

train_data = pd.read_csv('../input/dsn-ai-oau-july-challenge/train.csv')

test_data = pd.read_csv('../input/dsn-ai-oau-july-challenge/test.csv')
train_data.shape
train_data.head(10)
#All null values present in the train dataset

train_data.isnull().sum()
#print duplicates

print('Duplicated entries:',train_data.duplicated().sum())
#Value counts of product fat content

train_data['Product_Fat_Content'].value_counts()
#Finding the mean weight value based on product weight

data = train_data[['Product_Fat_Content','Product_Weight']]

data_grp = data.groupby(['Product_Fat_Content'])

data_grp.describe()
#Replacing null values of product weight with their respective means based on product fat content

LF = train_data['Product_Fat_Content'] == 'Low Fat'

NF = train_data['Product_Fat_Content'] == 'Normal Fat'

ULF = train_data['Product_Fat_Content'] == 'Ultra Low fat'



train_data.loc[LF,'Product_Weight'] = train_data.loc[LF,'Product_Weight'].fillna(train_data.loc[LF,'Product_Weight'].mean())

train_data.loc[NF,'Product_Weight'] = train_data.loc[NF,'Product_Weight'].fillna(train_data.loc[NF,'Product_Weight'].mean())

train_data.loc[ULF,'Product_Weight'] = train_data.loc[ULF,'Product_Weight'].fillna(train_data.loc[ULF,'Product_Weight'].mean())
#print first 10 rows of train dataset

train_data.head(10)
#confirming all null values for product weight have been filled

train_data.isnull().sum()
#drop ID values

train_data.drop(columns=['Supermarket_Identifier','Product_Supermarket_Identifier','Product_Identifier'],inplace=True)
train_data_copy = train_data.copy()
#fill supermarket missing values

train_data['Supermarket _Size'].fillna(method='bfill', inplace=True)
train_data
#all null values handled

train_data.isnull().sum()
#Group by product type and fat content

Grp_Type_Fat = train_data[:10].groupby(['Product_Type','Product_Fat_Content'])

Grp_Type_Fat.first()
#crosstab of supermarket size with location type

pd.crosstab(train_data['Supermarket _Size'],train_data['Supermarket_Location_Type'],margins=True)
#Create a pivot table

pd.pivot_table(train_data,index=['Supermarket_Opening_Year'], values='Product_Supermarket_Sales')
#Create another pivot table including the supermarket size, location and opening year

pd.pivot_table(train_data, index=['Supermarket_Opening_Year','Supermarket _Size','Supermarket_Location_Type'],values='Product_Supermarket_Sales',aggfunc=[np.mean, np.median, min, max, np.std])
fig_dims = (25, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x='Product_Type', y='Product_Price',ax=ax, data=train_data)
sns.lineplot(x='Product_Weight', y='Product_Price', data=train_data)
sns.lineplot(x='Supermarket_Opening_Year', y='Product_Supermarket_Sales',data=train_data)
sns.barplot(x='Product_Fat_Content', y='Product_Supermarket_Sales',data=train_data)
sns.barplot(x='Supermarket_Opening_Year', y='Product_Supermarket_Sales',data=train_data)
sns.boxplot(train_data['Product_Supermarket_Sales'], orient='vertical')
sns.lineplot(x='Supermarket_Opening_Year', y='Product_Price',data=train_data)
sns.catplot(data=train_data, x='Supermarket_Opening_Year', y='Product_Supermarket_Sales', row='Product_Fat_Content', kind='swarm', height=3, aspect=4)
sns.heatmap(train_data.corr(), annot = True, fmt = '.1g',vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm', linewidths = 2, linecolor = 'black')
train_data.hist()

plt.tight_layout()