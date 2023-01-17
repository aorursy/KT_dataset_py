import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

housing = pd.read_csv('../input/housing.csv')

housing.head()
housing.info()
housing.ocean_proximity.value_counts()
housing.describe()
sns.set()

housing.isna().sum().sort_values(ascending=True).plot(kind='barh',figsize=(10,7))#Quick peak into the missing columns values

#Let's deal with that later on the cleaning part with various methods !
housing.hist(bins=50,figsize=(20,15))#The bins parameter is used to custom the number of bins shown on the plots.

plt.show()
from sklearn.model_selection import train_test_split

train_, test_ = train_test_split(housing,test_size=0.2,random_state=1)
plotter = housing.copy()
sns.set()

plt.figure(figsize=(10,8))#Figure size

plt.scatter('longitude','latitude',data=plotter)

plt.ylabel('Latitudes')

plt.xlabel('Longitudes')

plt.title('Geographical plot of Lats/Lons')

plt.show()
sns.set()

plt.figure(figsize=(10,8))#Figure size

plt.scatter('longitude','latitude',data=plotter,alpha=0.1)

plt.ylabel('Latitudes')

plt.xlabel('Longitudes')

plt.title('Geographical plot of Lats/Lons')

plt.show()
plt.figure(figsize=(10,7))

plotter.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

        s=plotter["population"]/100, label="population", figsize=(15,8),

        c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,

    )

plt.legend()
corr_matrix=plotter.corr()

corr_matrix.median_house_value.sort_values(ascending=False)
from pandas import scatter_matrix

sns.set()

feat = ['median_house_value','median_income','total_rooms','housing_median_age']

scatter_matrix(plotter[feat],figsize=(15,8))
plt.figure(figsize=(12,7))

plt.scatter('median_income','median_house_value',data=plotter,alpha=0.1)

plt.xlabel('Median income')

plt.ylabel('Median house value')

plt.title('Linear correlation Median income/Median House value')
plotter['rooms_per_household']= plotter.total_rooms/housing.households
plotter.head()
corr_matrix1=plotter.corr()

corr=corr_matrix1.median_house_value.sort_values(ascending=False)

d= pd.DataFrame({'Column':corr.index,

                 'Correlation with median_house_value':corr.values})

d
#plotter.dropna(subset=["total_bedrooms"]) # option 1 

#plotter.drop("total_bedrooms", axis=1) # option 2 

#median = plotter["total_bedrooms"].median() # option 3 

#plotter["total_bedrooms"].fillna(median, inplace=True)

from sklearn.impute import SimpleImputer

imputer =SimpleImputer(strategy='median')#In this case its better to use the median to replace missing values
ft_data = plotter.drop('ocean_proximity',axis=1)
imputer.fit(ft_data)

imputer.statistics_ #Here's the median of every attribute in our data !
ft_data.total_bedrooms.median()
X = imputer.transform(ft_data)
ft_transformed = pd.DataFrame(X,columns=ft_data.columns)

ft_transformed.tail() #The missing values in total_bedrooms were replaced by the median value
obj_cols = housing.dtypes

obj_cols[obj_cols=='object']
sns.set(palette='Set2')

housing.ocean_proximity.value_counts().sort_values(ascending=True).plot(kind='barh',figsize=(10,7))

plt.legend()
from sklearn.preprocessing import OneHotEncoder

lab_encoder = OneHotEncoder()

cat_house = housing[['ocean_proximity']]

cat_enc = lab_encoder.fit_transform(cat_house)