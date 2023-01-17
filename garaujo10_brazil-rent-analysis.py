#Importing libs

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("darkgrid")

sns.set_palette("Dark2_r")
df1 = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv', index_col = 0)

df2 = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
print(df1.head())

print('------------------------------------------------------')

print(df2.head())
df = df2.copy()

print('Printing dataset`s info\n')

print(df.info())

print('-----------------------------------------------')

print('Checking the quantity of null values\n')

print(df.isna().sum())
df.head(10)
df[df['floor'].str.contains('\-')==True]
print('Checking the mode for each column')

df.mode()
print('Checking the values quantity\n')

print(df.floor.value_counts())
df['floor'].replace(to_replace=r'\-', value=1, regex=True, inplace = True)

df.head(10)
df[df['floor'].str.contains('\-')==True]
#Transform floor to integer

df = df.astype({'floor': 'int64'})

df.head(10)
df.describe().round(2)
ax = sns.boxplot(df['area'])

ax.figure.set_size_inches(20,6)

ax.set_title('Area', fontsize=20)

ax.set_xlabel('Area (m²)', fontsize=16)

ax;
ax = sns.distplot(df['area'])

ax.figure.set_size_inches(20,6)

ax.set_title('Area', fontsize=20)

ax.set_xlabel('Area (m²)', fontsize=16)

ax;
ax = sns.boxplot(df['rooms'])

ax.figure.set_size_inches(20,6)

ax.set_title('Rooms', fontsize=20)

ax.set_xlabel('Number of rooms', fontsize=16)

ax;
ax = sns.distplot(df['rooms'])

ax.figure.set_size_inches(20,6)

ax.set_title('Rooms', fontsize=20)

ax.set_xlabel('Number of rooms', fontsize=16)

ax;
ax = sns.boxplot(df['bathroom'])

ax.figure.set_size_inches(20,6)

ax.set_title('Bathrooms', fontsize=20)

ax.set_xlabel('Number of bathroom', fontsize=16)

ax;
ax = sns.distplot(df['bathroom'])

ax.figure.set_size_inches(20,6)

ax.set_title('Bathrooms', fontsize=20)

ax.set_xlabel('Number of bathroom', fontsize=16)

ax;


ax = sns.boxplot(df['parking spaces'])

ax.figure.set_size_inches(20,6)

ax.set_title('Parking spaces', fontsize=20)

ax.set_xlabel('Number of parking spaces', fontsize=16)

ax;
ax = sns.distplot(df['parking spaces'])

ax.figure.set_size_inches(20,6)

ax.set_title('Parking spaces', fontsize=20)

ax.set_xlabel('Number of parking spaces', fontsize=16)

ax;
ax = sns.boxplot(df['total (R$)'])

ax.figure.set_size_inches(20,6)

ax.set_title('Final price', fontsize=20)

ax.set_xlabel('Price (R$)', fontsize=16)

ax;
ax = sns.distplot(df['total (R$)'])

ax.figure.set_size_inches(20,6)

ax.set_title('Final price', fontsize=20)

ax.set_xlabel('Price (R$)', fontsize=16)

ax;
sns.pairplot(df);
ax = sns.pairplot(df, y_vars='total (R$)', x_vars=['area', 'rooms', 'bathroom', 'parking spaces', 'floor'], height=5, kind='reg')

ax;
#Area outliers

np.sort(df['area'].unique())[-20:]
df[df['area']>1000]
df.drop(df[df['area']>1000].index, inplace = True)
ax = sns.boxplot(df['area'])

ax.figure.set_size_inches(20,6)

ax.set_title('Area', fontsize=20)

ax.set_xlabel('Area (m²)', fontsize=16)

ax;
ax = sns.distplot(df['area'])

ax.figure.set_size_inches(20,6)

ax.set_title('Area', fontsize=20)

ax.set_xlabel('Area (m²)', fontsize=16)

ax;
ax = sns.distplot(df['rooms'])

ax.figure.set_size_inches(20,6)

ax.set_title('rooms', fontsize=20)

ax.set_xlabel('rooms', fontsize=16)

ax;
#Total outliers

print('Highest prices')

print(np.sort(df['total (R$)'].unique())[-50:])

print('---------------------------------------------------')

print('Total price describe')

print(df['total (R$)'].describe().round(2))
df.drop(df[df['total (R$)']>20000].index, inplace = True)
ax = sns.distplot(df['total (R$)'])

ax.figure.set_size_inches(20,6)

ax.set_title('Final price', fontsize=20)

ax.set_xlabel('Price (R$)', fontsize=16)

ax;
ax = sns.boxplot(df['total (R$)'])

ax.figure.set_size_inches(20,6)

ax.set_title('Final price', fontsize=20)

ax.set_xlabel('Price (R$)', fontsize=16)

ax;
#Checking the last pairplot line again

ax = sns.pairplot(df, y_vars='total (R$)', x_vars=['area', 'rooms', 'bathroom', 'parking spaces', 'floor'], height=5, kind='reg')

ax;
plt.figure(figsize=(10,10))

ax = sns.heatmap(df.corr())

ax.set_title('Correlation', fontsize=20)

ax;
df.city.unique()
city_test = pd.get_dummies(df) #Let's see what city is the most expensive

city_test.corr().head(15)
ax = sns.boxplot(data = df, x = 'city', y = 'total (R$)', orient = 'v')

ax.figure.set_size_inches(20,6)

ax.set_title('Final price per city', fontsize=20)

ax.set_xlabel('City', fontsize=16)

ax.set_ylabel('Price (R$)', fontsize=16)

ax;