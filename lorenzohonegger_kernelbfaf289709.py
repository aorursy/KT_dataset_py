# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## load data into dataframe

df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.info()
sns.heatmap(df.isna())
df.describe()
sns.distplot(df['number_of_reviews'], kde = False)
sns.pairplot(data = df, vars = ['number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count'])



## create dummy variable for room type

room_type = pd.get_dummies(df['room_type'], drop_first = True)

df['room_private'] = room_type['Private room'];  df['room_shared'] = room_type['Shared room'];

df = df.drop(['room_type'] , axis = 1)



print(df.head(3));
## create dummy variable for neighbourhood group

neigh_group = pd.get_dummies(df['neighbourhood_group'], drop_first = True)

print(neigh_group.head())

df['brooklyn'] = neigh_group['Brooklyn'];  df['manhattan'] = neigh_group['Manhattan'];

df['queens'] = neigh_group['Queens'];  df['staten_island'] = neigh_group['Staten Island'];

df = df.drop(['neighbourhood_group'] , axis = 1)

print(df.head(3));
sns.distplot(df['host_id'], kde = False)
sns.pairplot(df, vars = ["latitude", "longitude", "price", "host_id"], hue = "room_private")



sns.heatmap(df[['manhattan', 'price',  'calculated_host_listings_count' ,'availability_365', 'room_shared' ]].corr(), annot = True)



## select properties where price is below $1000

df = df[df.price < 1000]



X = df[['manhattan', 'calculated_host_listings_count' ,'availability_365', 'room_shared' ]]

y = df['price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))