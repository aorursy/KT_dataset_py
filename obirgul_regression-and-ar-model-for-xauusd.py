# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/xauusd/XAU_USD_last_2_years_data.csv')

df
df2 = pd.read_csv("../input/xauusd/CNY_USD_last_2_years_data.csv")

df2
df = df.drop(['Şimdi', 'Yüksek','Düşük','Fark %'], axis=1)

df = df.rename(index=str,columns={'Açılış':'Gold_Open','Tarih':'Date'})

df
df2 = df2.drop(['Şimdi', 'Yüksek','Düşük','Fark %'], axis=1)

df2 = df2.rename(index=str,columns={'Açılış':'Yuan_Open','Tarih':'Date'})

df2
df['Date'] = pd.to_datetime(df['Date'], format="%d.%m.%Y")

df = df.sort_values(by=['Date'], ascending=[True])

df.head(7)
df2['Date'] = pd.to_datetime(df2['Date'], format="%d.%m.%Y")

df2 = df2.sort_values(by=['Date'], ascending=[True])

df.head(7)
dates = pd.date_range(start=df.Date.min(), end=df.Date.max())
df2 = df2.set_index('Date').reindex(dates).fillna(method ='ffill', limit = 1).rename_axis('Date').reset_index()

df2["Yuan_Open"] = df2["Yuan_Open"].fillna(method ='bfill', limit = 2)

df2.describe()
df = df.set_index('Date').reindex(dates).fillna(method ='ffill', limit = 1).rename_axis('Date').reset_index()

df["Gold_Open"] = df["Gold_Open"].fillna(method ='bfill', limit = 2)

#pazar günleri pazartesiye, cumartesi günleri cumaya eşitlenmiştir. ??

df.describe()
df = pd.merge(df, df2)

df.head(20)
df.describe()
df.isnull().sum()
df.info()
df['Gold_Open'] = df['Gold_Open'].str.replace('.','')

df['Gold_Open'] = df['Gold_Open'].str.replace(',','.')

df['Gold_Open'] = df['Gold_Open'].astype(float)
df['Yuan_Open'] = df['Yuan_Open'].str.replace(',','.')

df['Yuan_Open'] = df['Yuan_Open'].astype(float)

df.head()
df['day_of_week'] = df['Date'].dt.day_name()

df.head(7)
plt.plot(df['Date'], df['Gold_Open'], color='green', marker='o', linestyle='dashed', linewidth=2)

plt.show()
plt.plot(df['Date'], df['Yuan_Open'], color='green', marker='o', linestyle='dashed', linewidth=2)

plt.show()
y=df.Gold_Open.values

x=df.drop(["Gold_Open","Date"],axis=1)

x.head()
x = pd.get_dummies(data = x, columns = ['day_of_week'], drop_first = False)

x.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)
from sklearn.linear_model import LinearRegression

from sklearn import metrics



lin_reg = LinearRegression()  

lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_test)



y_train_pred = lin_reg.predict(x_train)



predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

predicted.head(10)
plt.plot(y_train, color='green', marker='o', linestyle='dashed', linewidth=2)

plt.plot(y_train_pred, color='red', marker='o', linestyle='dashed', linewidth=2)

plt.show()
plt.plot(y_test, color='green', marker='o', linestyle='dashed', linewidth=2)

plt.plot(y_pred, color='red', marker='o', linestyle='dashed', linewidth=2)

plt.show()
predicted.plot(kind='bar',figsize=(10,8))

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
df.head()
# split dataset

X = df['Gold_Open'].values

train, test = X[1:len(X)-7], X[len(X)-7:]
from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error



# train autoregression

model = AR(train)

model_fit = model.fit()

print('Lag: %s' % model_fit.k_ar)

print('Coefficients: %s' % model_fit.params)
# make predictions

predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

for i in range(len(predictions)):

    print('predicted=%f, expected=%f' % (predictions[i], test[i]))

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)
# plot results

plt.plot(test)

plt.plot(predictions, color='red')

plt.show()