import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.head()
df.isnull().sum()
df.describe().transpose()
plt.figure(figsize=(12,8))
sns.distplot(df['price'])
sns.countplot(df['bedrooms'])
plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='sqft_living',data=df)
plt.figure(figsize=(12,8))
sns.boxplot(x='bedrooms',y='price',data=df)
plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='long',data=df)
plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='lat',data=df)
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')
df.sort_values('price',ascending=False).head(20)
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',
                data=non_top_1_perc,hue='price',
                palette='RdYlGn',edgecolor=None,alpha=0.2)
plt.figure(figsize=(12,8))
sns.boxplot(x='waterfront',y='price',data=df)
df.head()
df.info()
df = df.drop('id',axis=1)
df.head()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)
plt.figure(figsize=(12,10))

plt.subplot(2, 2, 1)
sns.boxplot(x='year',y='price',data=df)

plt.subplot(2, 2, 2)
sns.boxplot(x='month',y='price',data=df)
plt.figure(figsize=(12,10))

plt.subplot(2, 2, 1)
df.groupby('month').mean()['price'].plot()

plt.subplot(2, 2, 2)
df.groupby('year').mean()['price'].plot()

df = df.drop('date',axis=1)
df.columns
# https://i.pinimg.com/originals/4a/ab/31/4aab31ce95d5b8474fd2cc063f334178.jpg
# May be worth considering to remove this or feature engineer categories from it
df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)
df.head()
# could make sense due to scaling, higher should correlate to more value
df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
model = Sequential()

model.add(Dense(X_train.shape[1],activation='relu'))
model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))
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
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print_evaluate(y_train, y_train_pred, train=True)
print_evaluate(y_test, y_test_pred, train=False)