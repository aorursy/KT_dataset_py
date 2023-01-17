# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # graphs

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv', sep = ',')

df.head()
df.tail(5)
print(df.shape)
print(df.size)
list(df.columns)
df.dtypes
df.isnull().sum()
# Creating a map

import folium

from folium.plugins import MarkerCluster

HousesMap = folium.Map(location = [47.501143, -121.824001])
#creating a Marker for each point in df

mc = MarkerCluster()

for row in df.itertuples():

    mc.add_child(folium.Marker(location=[row.lat, row.long], popup = str(row.sqft_living) + '/' + str(row.sqft_lot)))

HousesMap.add_child(mc)
HousesMap.save('/kaggle/working/map.html')
df1=df[['price', 'bedrooms', 'bathrooms', 'sqft_living',

    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

    'lat', 'long', 'sqft_living15', 'sqft_lot15']]

h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)

sns.despine(left=True, bottom=True)

[x.title.set_size(12) for x in h.ravel()];

[x.yaxis.tick_left() for x in h.ravel()];
sns.set(style="whitegrid", font_scale=1)
f, axes = plt.subplots(1, 2,figsize=(15,5))

sns.boxplot(x=df['bedrooms'],y=df['price'], ax=axes[0])

sns.boxplot(x=df['floors'],y=df['price'], ax=axes[1])

sns.despine(left=True, bottom=True)

axes[0].set(xlabel='Bedrooms', ylabel='Price')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].set(xlabel='Floors', ylabel='Price')



f, axe = plt.subplots(1, 1,figsize=(12.18,5))

sns.despine(left=True, bottom=True)

sns.boxplot(x=df['bathrooms'],y=df['price'], ax=axe)

axe.yaxis.tick_left()

axe.set(xlabel='Bathrooms / Bedrooms', ylabel='Price');
f, axes = plt.subplots(1, 2,figsize=(15,5))

sns.boxplot(x=df['waterfront'],y=df['price'], ax=axes[0])

sns.boxplot(x=df['view'],y=df['price'], ax=axes[1])

sns.despine(left=True, bottom=True)

axes[0].set(xlabel='Waterfront', ylabel='Price')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].set(xlabel='View', ylabel='Price')



f, axe = plt.subplots(1, 1,figsize=(12.18,5))

sns.boxplot(x=df['grade'],y=df['price'], ax=axe)

sns.despine(left=True, bottom=True)

axe.yaxis.tick_left()

axe.set(xlabel='Grade', ylabel='Price');
df['date'] = pd.to_datetime(df['date'])

df = df.set_index('id')

df.price = df.price.astype(int)

df.bathrooms = df.bathrooms.astype(int)

df.floors = df.floors.astype(int)

df.head(5)
df["house_age"] = df["date"].dt.year - df['yr_built']

df['renovated'] = df['yr_renovated'].apply(lambda yr: 0 if yr == 0 else 1)



df = df.drop('date', axis=1)

df = df.drop('yr_renovated', axis=1)

df = df.drop('yr_built', axis=1)

df.head(5)
df.describe().transpose()
correlation = df.corr(method='pearson')

columns = correlation.nlargest(10, 'price').index

columns
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
plt.figure(figsize=(12,8))

sns.scatterplot(x='price',y='sqft_living',data=df)
plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',data=df,hue='price')
df.sort_values('price',ascending=False).head(20)
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',

                data=non_top_1_perc,hue='price',

                palette='RdYlGn',edgecolor=None,alpha=0.2)
df.shape
from scipy import stats

z = np.abs(stats.zscore(df))

print(z)
threshold = 3

print(np.where(z > 3))
df_o = df[(z < 3).all(axis=1)]
df_o.shape
import keras

from keras import metrics

from keras import regularizers

from keras.optimizers import Adam, RMSprop

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from keras.utils import plot_model

from keras.models import load_model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D
X = df.drop('price',axis=1)

y = df['price']
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler





X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)



scaler = MinMaxScaler()



X_train= scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



print(X_train.shape)

print(X_test.shape)
from sklearn import metrics



def print_evaluate(true, predicted, train=True):  

    mae = metrics.mean_absolute_error(true, predicted)

    mse = metrics.mean_squared_error(true, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))

    r2_square = metrics.r2_score(true, predicted)

    if train:

        print("========Training Result=======")

        print('MAE: ', mae)

        print('MSE: ', mse)

        print('RMSE: ', rmse)

        print('R2 Square: ', r2_square)

    elif not train:

        print("=========Testing Result=======")

        print('MAE: ', mae)

        print('MSE: ', mse)

        print('RMSE: ', rmse)

        print('R2 Square: ', r2_square)
model = Sequential()



model.add(Dense(X_train.shape[1],activation='relu'))

model.add(Dense(32,activation='relu'))

# model.add(Dropout(0.2))



model.add(Dense(64,activation='relu'))

# model.add(Dropout(0.2))



model.add(Dense(128,activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1))



model.compile(optimizer=Adam(0.001), loss='mse')
r = model.fit(X_train, y_train.values,

              validation_data=(X_test,y_test.values),

              batch_size=128,

              epochs=500)
plt.figure(figsize=(10, 6))



plt.plot(r.history['loss'], label='loss')

plt.plot(r.history['val_loss'], label='val_loss')

plt.legend()
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)



print_evaluate(y_train, y_train_pred, train=True)

print_evaluate(y_test, y_test_pred, train=False)
df['price'].mean()
prediction = model.predict(X_test)
a = np.empty(6484)

b = np.arange(1, 6484, 1)

ind = np.arange(len(a))

np.put(a, ind, b)

print(a)
plt.plot(a, y_test, 'r-', label='actual')

plt.plot(a, prediction, 'b-', label='predicted')

plt.legend(loc='best')

plt.show()
X_o = df_o.drop('price',axis=1)

y_o = df_o['price']
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler





X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_o, y_o, test_size=0.3, random_state=101)



scaler = MinMaxScaler()



X_train_o = scaler.fit_transform(X_train_o)

X_test_o = scaler.transform(X_test_o)



print(X_train_o.shape)

print(X_test_o.shape)
model2 = Sequential()



model2.add(Dense(X_train.shape[1],activation='relu'))

model2.add(Dense(32,activation='relu'))

# model.add(Dropout(0.2))



model2.add(Dense(64,activation='relu'))

# model.add(Dropout(0.2))



model2.add(Dense(128,activation='relu'))

model2.add(Dense(256, activation='relu'))

model2.add(Dense(512, activation='relu'))

model2.add(Dense(1024, activation='relu'))

model2.add(Dropout(0.2))

model2.add(Dense(1))



model2.compile(optimizer=Adam(0.001), loss='mse')
r_o = model2.fit(X_train_o, y_train_o.values,

              validation_data=(X_test_o, y_test_o.values),

              batch_size=128,

              epochs=500)
plt.figure(figsize=(10, 6))



plt.plot(r_o.history['loss'], label='loss')

plt.plot(r_o.history['val_loss'], label='val_loss')

plt.legend()
y_train_o_pred = model2.predict(X_train_o)

y_test_o_pred = model2.predict(X_test_o)



print_evaluate(y_train_o, y_train_o_pred, train=True)

print_evaluate(y_test_o, y_test_o_pred, train=False)
df_o['price'].mean()
prediction = model2.predict(X_test)
a = np.empty(6484)

b = np.arange(1, 6484, 1)

ind = np.arange(len(a))

np.put(a, ind, b)

print(a)
plt.plot(a, y_test, 'r-', label='actual')

plt.plot(a, prediction, 'b-', label='predicted')

plt.legend(loc='best')

plt.show()