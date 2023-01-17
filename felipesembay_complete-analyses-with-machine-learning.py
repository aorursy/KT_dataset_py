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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
sales = pd.read_csv('../input/seattle-house-sales-prices/house_sales.csv')

sales



sales.count()
pd.value_counts(sales['bedrooms'])
sales['view']

sales.sort_values(by='view', ascending=False)
sales.loc[(sales['bedrooms']==3) & (sales['bathrooms'] > 2) & (sales['price'] < 200000) & (sales['yr_built']> 1990)]
sales.plot(x='bathrooms',y='price',kind='scatter',color='y')
sns.lmplot(x='bathrooms',y='price',data=sales)
sns.lmplot(x='sqft_living',y='price',data=sales)
sns.lmplot(x='bedrooms',y='price',data=sales)
sns.lmplot(x='view',y='price',data=sales)
sns.heatmap(sales.corr(),cmap='coolwarm',annot=True)

plt.rcParams['figure.figsize'] = (15.5, 10.5)
X = sales[['sqft_living','floors','waterfront','grade','sqft_lot', 'bedrooms', 'bathrooms', 'view']]

y = sales['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
lm.score(X_train, y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)

print(y_test,predictions)
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
plt.figure(figsize=(25,8))



plt.plot(range(0, y_test.shape[0]), y_test, marker='+')

plt.plot(range(0, predictions.shape[0]), predictions, marker='o')
import folium

from folium import plugins
lat = sales['lat'][:500].values

long = sales['long'][:500].values



mapa = folium.Map(location=[47.608013,-122.335167],zoom_start=4)



for la,lo in zip(lat,long):

    folium.Marker([la, lo]).add_to(mapa)



mapa
coordenadas = []

lat = sales['lat'][:500].values

long = sales['long'][:500].values



mapa = folium.Map(location=[47.608013,-122.335167],zoom_start=4)



for la,lo in zip(lat,long):

    coordenadas.append([la,lo])



mapa.add_child(plugins.HeatMap(coordenadas))

mapa