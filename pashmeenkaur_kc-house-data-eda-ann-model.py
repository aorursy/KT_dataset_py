# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/kc-housesales-data/kc_house_data.csv')
df.head()
df.info()
df['price'].describe()
plt.figure(figsize=(12,6))
sns.distplot(df['price'])
df.corr()['price'].sort_values(ascending=False).drop('price').plot(kind='bar')
sns.scatterplot(y='price',x='sqft_living',data=df)
sns.countplot('bedrooms',data=df)
plt.figure(figsize=(10,12))
sns.boxplot(x='bedrooms',y='price',data=df)
plt.figure(figsize=(10,9))
sns.boxplot(x=df['grade'],y='price',data=df)
sns.scatterplot(y='lat',x='price',data=df)
sns.scatterplot(y='long',x='price',data=df)
plt.figure(figsize=(12,10))
sns.scatterplot(y='lat',x='long',data=df,hue='price',edgecolor=None,alpha=0.4,palette='RdYlGn')
0.05*len(df)
new_99_df=df.sort_values('price',ascending=False).iloc[1080:]
len(new_99_df)
plt.figure(figsize=(12,10))
sns.scatterplot(y='lat',x='long',data=new_99_df,hue='price',edgecolor=None,alpha=0.9,palette='RdYlGn')
sns.boxplot(x='waterfront',y='price',data=new_99_df)
sns.countplot(df['view'])
df.info()
df['date'].head()
df['date']=pd.to_datetime(df['date'])
df['date'].head()
df.columns
df['month_sold']=df['date'].apply(lambda date: date.month)
df['year_sold']=df['date'].apply(lambda date: date.year)
df['month_sold'].head()
df['year_sold']
sns.countplot(df['month_sold'])
sns.countplot(df['year_sold'])
df['zipcode'].value_counts()
df=df.drop(['date','id','zipcode'],axis=1)
df.columns
X=df.drop('price',axis=1).values
y=df['price'].values
type(X)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)
x_train.shape
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=128,epochs=400)
loss=pd.DataFrame(model.history.history)
loss.plot()
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score

predictions=model.predict(x_test)

print("MAE:",mean_absolute_error(y_test,predictions))
print("MSE:",mean_squared_error(y_test,predictions))
print("RMSE:",np.sqrt(mean_squared_error(y_test,predictions)))
explained_variance_score(y_test,predictions)
plt.figure(figsize=(10,5))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')
