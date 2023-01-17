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
import seaborn as sns

import matplotlib.pyplot as plt
import cufflinks as cf
%matplotlib inline

cf.go_offline()
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.isnull().sum()
plt.figure(figsize=(20,20))

sns.heatmap(df.isnull(),cmap='plasma')
df['date'] = pd.to_datetime(df['date'])
df.head()
df.dtypes
df['year'] = df['date'].apply(lambda date:date.year)

df['month'] = df['date'].apply(lambda date:date.month)
df.drop('date',inplace=True,axis=1)
df.head()
df.columns
sns.set_style('whitegrid')

plt.figure(figsize=(12,9))

sns.distplot(df['price'])
df.describe()
#df.iplot(y='price',x='yr_built')
df.drop('zipcode',axis=1,inplace=True)
df.columns
df['yr_renovated'].nunique()
df.drop('yr_renovated',axis=1,inplace=True)
df.columns
X = df.drop('price',axis=1).values

y = df['price'].values
X
y
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=101)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Activation

from tensorflow.keras.optimizers import Adam
model = Sequential()



model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(19,activation='relu'))

model.add(Dense(1))



model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)
losses = pd.DataFrame(model.history.history)
losses.plot()
predictions = model.predict(X_test)
print(mean_absolute_error(y_test,predictions))

print('\n')

print(mean_squared_error(y_test,predictions))
from sklearn.metrics import explained_variance_score
print(explained_variance_score(y_test,predictions))
plt.scatter(y_test,predictions)



plt.plot(y_test,y_test,'r')
errors = y_test - predictions
sns.distplot(errors)