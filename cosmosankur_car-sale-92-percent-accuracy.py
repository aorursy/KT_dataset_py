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
import pandas as pd
df = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')











df.head()
df.isnull().any()
df.isnull().sum()
import seaborn as sns

sns.heatmap(df.isnull())
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.heatmap(df.corr())
plt.scatter(df['Present_Price'],df['Selling_Price'])

plt.xlabel('Present Price')

plt.ylabel('Selling Price')
plt.scatter(df['Kms_Driven'],df['Selling_Price'])
#list of numerical values

numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']



print('number of numerical variables', len(numerical_features))

print('total features ',len(df.columns))



df[numerical_features].head()
df.groupby('Year')['Selling_Price'].median().plot()

plt.xlabel('Year')

plt.ylabel('Selling Price')
df['year_diff'] = 2020 - df['Year']

df = df.drop('Year',axis=1)

df.head()
plt.scatter(df['year_diff'],df['Selling_Price'])

plt.xlabel('Year Difference')

plt.ylabel('Selling Price')
data = df.copy()

data.groupby(df['Owner'])['Selling_Price'].median().plot.bar()

plt.xlabel('No of owners previously')

plt.ylabel('Selling price')
continous_feature = [df['Kms_Driven'],df['year_diff'],df['Present_Price']]
df['Kms_Driven'].hist(bins=25)
df['year_diff'].hist(bins=25)
df['Present_Price'].hist(bins=25)


df['Kms_Driven'] = np.log(df['Kms_Driven'])

df['Present_Price'] = np.log(df['Present_Price'])
df['Kms_Driven'].hist(bins=25)
df['Present_Price'].hist(bins=25)
categorical_features = [feature for feature in df.columns if df[feature].dtypes=='O']
df[categorical_features].head()
df['Car_Name'].value_counts()

df['Fuel_Type'].value_counts()
df['Seller_Type'].value_counts()
df['Transmission'].value_counts()
for feature in categorical_features:

    data = df.copy()

    data.groupby(feature)['Selling_Price'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('Sale Price')

    plt.show()
df = df.drop('Car_Name',axis = 1)
df.head()
final_dataset = df
final_dataset = pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset.columns
x = final_dataset[['Present_Price', 'Kms_Driven', 'Owner', 'year_diff',

       'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual',

       'Transmission_Manual']]

y = final_dataset['Selling_Price']
x.head()
y.head()
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(x_train,y_train)

print(linear_model.score(x_test,y_test))
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()

tree.fit(x_train,y_train)

print(tree.score(x_test,y_test))
from sklearn.ensemble import RandomForestRegressor

forest =  RandomForestRegressor()

forest.fit(x_train,y_train)

print(forest.score(x_test,y_test))
from sklearn.ensemble import GradientBoostingRegressor

grad_boost = GradientBoostingRegressor()

grad_boost.fit(x_train,y_train)

print(grad_boost.score(x_test,y_test))
import xgboost

from xgboost import XGBRegressor

x_model = XGBRegressor()

x_model.fit(x_train,y_train)

print(x_model.score(x_test,y_test))