# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/housesalesprediction/kc_house_data.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
# first let us see if we have any missing data in each columns
data.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x= data['bedrooms'])
plt.figure(figsize=(10,5))
sns.distplot(data['price'])


sns.scatterplot(x=data['bedrooms'], y = data['price'])
# now let us see the correlation of our target with other values 
data.corr()['price'].sort_values()
# we see "sqft_living" has a strong correlation with price 
sns.scatterplot(x= data['sqft_living'], y =data['price'])
# in our data we have "lattitude" and "longitude" which give the house position in
# king country, USA
plt.figure(figsize=(5,8))
sns.scatterplot(x= data['long'],y =data['lat'], hue = data['price'])

# because of those extream points we are not getting a good 
# color distribution. so now let us drop those extreame points
data.sort_values('price',ascending=False).head(20)
len(data)

non_top_1_perc = data.sort_values('price',ascending=False).iloc[216:]
# let us plot it, to get more color distribution
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',
                data=non_top_1_perc,hue='price',
                palette='RdYlGn',edgecolor=None,alpha=0.2)
#Other Features
sns.boxplot(x='waterfront',y='price',data=data)
data.head()
data.info()
# now let us drop some featuers which is not very informatic
df = data.drop('id',axis=1)
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].apply(lambda date:date.month)
data['year'] = data['date'].apply(lambda date:date.year)
sns.boxplot(x='year',y='price',data=data)
sns.boxplot(x='month',y='price',data=data)
data.groupby('month').mean()['price'].plot()
data.groupby('year').mean()['price'].plot()
data = data.drop('date',axis=1)
data.columns

data['zipcode'].value_counts()
data = data.drop('zipcode',axis=1)
# could make sense due to scaling, higher should correlate to more value
data['yr_renovated'].value_counts()
data['sqft_basement'].value_counts()
X = df.drop('price',axis=1)
y = df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')