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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.head(10)
# Explore Data
df.isnull().sum()
df.describe().transpose()
plt.figure(figsize=(20,8))
sns.distplot(df["price"])
sns.countplot(df["bedrooms"])
# price between 0 - 2M and bedrooms 2 - 5
df.corr()["price"].sort_values(ascending=False)
plt.figure(figsize=(20,8))
sns.scatterplot(x="price",y="sqft_living",data=df)
plt.figure(figsize=(20,8))
sns.scatterplot(x="long",y="lat",data=df,hue="price",palette="RdYlGn",alpha=0.2,edgecolor=None)
bottom_99 = df.sort_values("price",ascending=False).iloc[216:]
plt.figure(figsize=(20,8))
sns.scatterplot(x="long",y="lat",data=bottom_99,hue="price",palette="RdYlGn",alpha=0.2,edgecolor=None)
#from the coordinates we can see that water side houses are more expenseive as natural. 
sns.boxplot(x="waterfront",y="price",data=bottom_99)
# so far we did a corraletion analysis, inspect lat/long - price relationship. exclude most expensive %1 of the houses.
df.head(10)
# what we do is to drop unusable columns and transform remaining ones as necessary. 
df = df.drop("id",axis=1)
df["date"]
df["date"] = pd.to_datetime(df["date"])
df["date"]
df["year"] = df["date"].apply(lambda date: date.year)
df["month"] = df["date"].apply(lambda date: date.month)
df = df.drop("date",axis=1)
plt.figure(figsize=(20,8))
sns.boxenplot(x="month",y="price",data=df)
plt.figure(figsize=(20,8))
df.groupby("month").mean()["price"].plot()
plt.figure(figsize=(20,8))
df.groupby("year").mean()["price"].plot()
#Also zipcode requires domain experince about the area itself. Normally we can categorize them.
#However, it wiil create 70 category which is too much to handle. 
#But if you are familiar with the area you can group them, and reduce category bin size. 
#I prefer to drop that column too. (Accept Bias on my model.)
df = df.drop("zipcode",axis=1)
df.head(10)
df["yr_renovated"].value_counts()
#Here we can say most of the data is equal to "0" so it is not a good variable in the context of quality
#It is possible to think that the building has not been renovated,that's why it is "0". 
#For example basement sqft value has also many "0" but you can think like there is no basement.
#So lets begin
X = df.drop("price",axis=1).values
y = df["price"].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(19,activation ="relu"))
model.add(Dense(19,activation ="relu"))
model.add(Dense(19,activation ="relu"))
model.add(Dense(19,activation ="relu"))

model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=250)
loss_df = pd.DataFrame(model.history.history)
#check if we overfit on test data and also see optimum epochs
loss_df.plot()
preds = model.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
mean_absolute_error(y_test,preds)
#This not mean anything if don't compare with the actual data, price avg value is 540K and the error 104K
#We are %20 percent wrong.
explained_variance_score(y_test,preds)
plt.scatter(y_test,preds)
plt.plot(y_test,y_test,"r")
#lets do everything with bottom_99 data set.But we re-define from our transformed and explored data df
bottom_99 = df.sort_values("price",ascending=False).iloc[216:]
X1 =bottom_99.drop("price",axis=1).values
y1 = bottom_99["price"].values
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)
scaler1  =  MinMaxScaler()
X1_train =  scaler1.fit_transform(X1_train)
X1_test  =  scaler1.transform(X1_test)
model1 = Sequential()
model1.add(Dense(19,activation ="relu"))
model1.add(Dense(19,activation ="relu"))
model1.add(Dense(19,activation ="relu"))
model.add(Dense(19,activation ="relu"))

model1.add(Dense(1))

model1.compile(optimizer="adam",loss="mse")
model1.fit(x=X1_train,y=y1_train,validation_data=(X1_test,y1_test),batch_size=64,epochs=500)
loss_bott99 = pd.DataFrame(model1.history.history)
loss_bott99.plot()
preds1 = model1.predict(X1_test)
mean_absolute_error(y1_test,preds1)
explained_variance_score(y1_test,preds1)
plt.scatter(y1_test,preds1)
plt.plot(y1_test,y1_test,"r")
#So we get worse around %30. So we need to optimize our model. Try different epochs and batch sizes, but we may need to update our data set, make more cleaning
#Also find the optimum neuron structure. Will be continued. 