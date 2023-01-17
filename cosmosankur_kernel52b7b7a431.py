# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/delhi-house-price-prediction/MagicBricks.csv')
df.head()
df.isnull().sum()
import seaborn as sns

sns.heatmap(df.isnull())
df.head(50)
df.shape
df = df[df.loc[:]!=0].dropna()

df.isnull().any(axis=0)
df.shape
import seaborn as sns

sns.heatmap(df.isnull())
sns.heatmap(df.corr())
#list of numerical values

numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']



print('number of numerical variables', len(numerical_features))

print('total features ',len(df.columns))



df[numerical_features].head()
import numpy as np

import matplotlib.pyplot as plt

plt.scatter(df['Area'],df['Price'])

plt.xlabel('Area')

plt.ylabel('Price')
plt.scatter(df['Per_Sqft'],df['Price'])
df['Area'].hist(bins=25)
df['Per_Sqft'].hist(bins=25)
df['Area'] = np.log(df['Area'])

df['Per_Sqft'] = np.log(df['Per_Sqft'])
df['Area'].hist(bins=25)
df['Per_Sqft'].hist(bins=25)
data = df.copy()

data.groupby(df['BHK'])['Price'].median().plot.bar()

plt.xlabel('BHK')

plt.ylabel(' price(in  tenlakhs)')

data = df.copy()

data.groupby(df['Bathroom'])['Price'].median().plot.bar()

plt.xlabel('Number of Bathroms')

plt.ylabel(' price(in ten lakhs)')
data = df.copy()

data.groupby(df['Parking'])['Price'].median().plot.bar()

plt.xlabel('Number of Parking')

plt.ylabel(' price(in ten lakhs)')
categorical_features = [feature for feature in df.columns if df[feature].dtypes=='O']
df[categorical_features].head()
df['Furnishing'].value_counts()
df['Status'].value_counts()
df['Transaction'].value_counts()
df['Type'].value_counts()
df['Locality'].value_counts()
df = df.drop('Locality',axis = 1)
df.head()
df = pd.get_dummies(df)
df.head()
x = df[['Area', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft',

       'Furnishing_Furnished', 'Furnishing_Semi-Furnished',

       'Furnishing_Unfurnished', 'Status_Almost_ready', 'Status_Ready_to_move',

       'Transaction_New_Property', 'Transaction_Resale', 'Type_Apartment',

       'Type_Builder_Floor']]

y = df['Price']
from sklearn.model_selection import train_test_split

x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size = 0.2,random_state=101)
from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(x_train,y_train)

print(linear.score(x_test,y_test))
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()

tree.fit(x_train,y_train)

print(tree.score(x_test,y_test))
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(x_train,y_train)

print(forest.score(x_test,y_test))
from sklearn.ensemble import GradientBoostingRegressor

grad = GradientBoostingRegressor()

grad.fit(x_train,y_train)

print(grad.score(x_test,y_test))
import xgboost

from xgboost import XGBRegressor

xboost = XGBRegressor()

xboost.fit(x_train,y_train)

print(xboost.score(x_test,y_test))