import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import collections

from sklearn.metrics import r2_score

import re

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor 

%matplotlib inline
df_train = pd.read_csv("/kaggle/input/used-cars-price-prediction/train-data.csv")
df_train.head()
df_train.isna().sum()
(df_train.isnull().sum() / len(df_train)) * 100
df_train = df_train.rename(columns = {'Unnamed: 0': 'id'})
df_train.isna().sum()
df_train.groupby('Seats')['id'].nunique()
df_train['Seats'].mode()
df_train["Seats"].fillna(value = 5.0, inplace=True)

df_train.Seats[df_train.Seats == 0.0] = 5.0

df_train.isna().sum()
df_train.groupby('Mileage')['id'].nunique()
df_train.Mileage[df_train.Mileage == '0.0 kmpl'] = np.nan

df_train['Mileage'] = df_train['Mileage'].apply(lambda x: re.sub(r'(\d+\.\d+)\s(kmpl|km\/kg)', 

                                                                 r'\1', str(x)))

df_train['Mileage'] = df_train['Mileage'].astype(float)

df_train['Mileage'].mode()
df_train['Mileage'].fillna(value = 17.0, inplace = True)

df_train.isna().sum()
df_train.groupby('Engine')['id'].nunique()
df_train['Engine'] = df_train['Engine'].apply(lambda x: re.sub(r'(\d+)\s(CC)', r'\1', str(x)))

df_train['Engine'] = df_train['Engine'].astype(float)

df_train['Engine'].mode()
df_train['Engine'].fillna(value = 1197.0, inplace = True)

df_train.isna().sum()
df_train['Power'] = df_train['Power'].str.split(' ').str[0]

# including nan rows there is data in this column of 'null' value

df_train.Power[df_train.Power == 'null'] = np.NaN

df_train['Power'].isnull().sum()
df_train['Power'] = df_train['Power'].astype(float)

df_train['Power'].mode()
df_train['Power'].fillna(value = 74, inplace = True)

df_train.isna().sum()
df_train['Name'] = df_train['Name'].str.split(' ').str[0]

df_train.groupby('Name')['id'].nunique()
df_train.Name[df_train.Name == 'Isuzu'] = 'ISUZU'
del df_train['New_Price']
dataset = df_train.copy()
del df_train['id']
df_train.dtypes
df_train['Year'] = df_train['Year'].astype(float)

df_train['Kilometers_Driven'] = df_train['Kilometers_Driven'].astype(float)
df_train['Price_log'] = np.log1p(df_train['Price'].values)

del df_train['Price']
df_train = pd.get_dummies(df_train, drop_first = True)
X = df_train.drop(columns = ['Price_log'], axis = 1)

y = df_train.iloc[:, 6].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
regressor_1 = LinearRegression()

regressor_1.fit(X_train, y_train)
y_pred_1 = regressor_1.predict(X_test)
regressor_1.score(X_test,y_test)
regressor_2 = RandomForestRegressor(random_state = 0)

regressor_2.fit(X_train, y_train)
y_pred_2 = regressor_2.predict(X_test)
regressor_2.score(X_test,y_test)
regressor_3 = DecisionTreeRegressor(random_state = 0)

regressor_3.fit(X_train, y_train)
y_pred_3 = regressor_3.predict(X_test)
regressor_3.score(X_test, y_test)
plt.style.use('ggplot')

colors = ['#FF8C73','#66b3ff','#99ff99','#CA8BCA', '#FFB973', '#89DF38', '#8BA4CA', '#ffcc99', 

          '#72A047', '#3052AF', '#FFC4C4']
plt.figure(figsize = (10,8))

bar1 = sns.countplot(dataset['Year'])

bar1.set_xticklabels(bar1.get_xticklabels(), rotation = 90, ha = 'right')

plt.title('Count year wise', size = 24)

plt.xlabel('Year', size = 18)

plt.ylabel('Count', size = 18)

plt.show()
plt.figure(figsize = (5,5))

sns.countplot(dataset['Fuel_Type'])

plt.title('Types of Fuel and count', size = 24)

plt.tight_layout()

plt.show()
plt.figure(figsize = (6,6))

plt.pie(dataset['Location'].value_counts(), startangle = 90, autopct = '%1.1f%%', colors = colors, 

        labels = dataset['Location'].unique())

centre_circle = plt.Circle((0,0),0.80,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.tight_layout()

plt.show()
plt.figure(figsize = (5,5))

sns.countplot(dataset['Transmission'])

plt.title('Types of transmission', size = 24)

plt.tight_layout()

plt.show()