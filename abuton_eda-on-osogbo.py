# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# for Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# To ignore Unnessary warnings

import warnings

warnings.filterwarnings('ignore')

# set the style to use for plotting

plt.style.use('ggplot')
# import diffrent algorithms. This will be use to establish a baseline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
df = pd.read_csv('/kaggle/input/aiosogbo-certification-competition/Train.csv')
# check the first 5 rows

df.head()
# checking the features (columns) names of the data

df.columns
df.shape
df.info()
df.isnull().sum()
numeric_features = df.select_dtypes(include=['int64', 'float64'])

categorical_features = df.select_dtypes(include='object')

print('Numeric Columns are {}'.format(numeric_features.columns))

print('-----------'*10)

print('Categorical Columns are {}'.format(categorical_features.columns))
# Our target is continous, so we can get the distribution using a histogram



plt.figure

plt.hist(df['Item_Outlet_Sales'], bins=50)

plt.xlabel('Item_Outlet_Sales')

plt.ylabel('count')

plt.title('Histogram of Item_Outlet_Sales')

plt.show()
# independent Variables (Numeric variables)

# histogram helps us to visualize the distribution of the variable

Item_Weight = df['Item_Weight']

fig, ax = plt.subplots()

ax.hist(Item_Weight.dropna(), color='blue', bins=50, alpha=0.9)

plt.xlabel('Item_Weight')

plt.ylabel('count')

plt.title('Histogram of Item_Weight')
Item_Visibility = df['Item_Visibility']

fig, ax = plt.subplots()

ax.hist(Item_Visibility.dropna(), color='green', bins=80, alpha=0.9)

plt.xlabel('Item_Visibility')

plt.ylabel('count')

plt.title('Histogram of Item_Visibility')
Item_MRP = df['Item_MRP']

fig, ax = plt.subplots()

ax.hist(Item_MRP.dropna(), color='red', bins=90, alpha=0.9)

plt.xlabel('Item_MRP')

plt.ylabel('count')

plt.title('Histogram of Item_MRP')
df['Item_Fat_Content'].value_counts().plot(kind='bar')



plt.xlabel('Item_Fat_Content')

plt.ylabel('count')

plt.title('Histogram of Item_Fat_Content')
df.Item_Fat_Content[df['Item_Fat_Content'] == 'LF'] = 'Low Fat'

df.Item_Fat_Content[df['Item_Fat_Content'] == 'low fat'] = 'Low Fat'



# for regular

df.Item_Fat_Content[df['Item_Fat_Content'] == 'reg'] = 'Regular'



# plot again



df['Item_Fat_Content'].value_counts().plot(kind='bar')



plt.xlabel('Item_Fat_Content')

plt.ylabel('count')

plt.title('Bar Chart of Item_Fat_Content')

# plot for Item_Type



df['Item_Type'].value_counts().plot(kind='bar')



plt.xlabel('Item_Fat_Content')

plt.ylabel('count')

plt.title('Bar Chart of Item_Fat_Content')

# plot for Outlet_Identifier



df['Outlet_Identifier'].value_counts().plot(kind='bar')
# plot for Outlet_Size



df['Outlet_Size'].value_counts().plot(kind='bar')

# plot for Establishment_Year

df.Outlet_Establishment_Year.value_counts().plot(kind = 'bar')
# plot for Outlet_Type

df['Outlet_Type'].value_counts().plot(kind='bar')
# Item_Weight vs Item_Outlet_Sales

plt.scatter(df['Item_Weight'], df['Item_Outlet_Sales'], c='violet', alpha=0.3, marker='.')

plt.xlabel('Item_Weight'), plt.ylabel('Item_Outlet_Sales'), plt.title('Item_Weight vs Item_Outlet_Sales')



######################## Item_Outlet_Sales is spread well across the entirerang
# Item_Visibility vs Item_Outlet_Sales

plt.scatter(df['Item_Visibility'], df['Item_Outlet_Sales'], c='violet', alpha=0.3, marker='.')

plt.xlabel('Item_Visibility'), plt.ylabel('Item_Outlet_Sales'), plt.title('Item_Visibility vs Item_Outlet_Sales')

 # Item_MRP vs Item_Outlet_Sales

plt.scatter(df['Item_MRP'], df['Item_Outlet_Sales'], c='violet', alpha=0.3, marker='.')



plt.xlabel('Item_MRP'), plt.ylabel('Item_Outlet_Sales'), plt.title('Item_MRP vs Item_Outlet_Sales')
sns.violinplot(df['Item_Type'], df['Item_Outlet_Sales'])

plt.xticks(rotation=90)

sns.violinplot(df['Item_Fat_Content'], df['Item_Outlet_Sales'])

plt.xticks(rotation=90)
sns.violinplot(df['Outlet_Size'], df['Item_Outlet_Sales'])

plt.xticks(rotation=40)
df.isnull().sum()
# fill int or float data type with median and fill categorical data type with mode.

# Note this is optional you can use what you see best

df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())

# remember Item_Visibility as lots of 0 values which is not possible

plt.hist(df['Item_Visibility'], bins=70, color='grey')

plt.show()
# let replace the zeroes and plot again to see the changes

zero_index = df['Item_Visibility'] == 0



df['Item_Visibility'] = df['Item_Visibility'].replace(0, np.median(df.Item_Visibility))

plt.hist(df['Item_Visibility'], bins=70, color='grey')

plt.show()