# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib as plt
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
weather = pd.read_csv("../input/weatherHistory.csv")
weather
weather.head(10)
weather.groupby('Summary').mean()["Apparent Temperature (C)"].plot(kind='bar')
weather[weather["Summary"] == "Dry"].mean()
def convert_summary(col):
    return len(col)

# Need to find the Apparent temperature when humidity given
weather_temp = weather[["Humidity","Apparent Temperature (C)"]]
#weather_temp["Summary"] = weather["Summary"].apply(convert_summary) 
weather_temp.head(12)
weather.groupby('Summary').mean()["Apparent Temperature (C)"].plot(kind='bar')
dummies = pd.get_dummies(weather["Summary"])
dummies.head(122)
weather_temp2 = pd.concat([weather_temp,dummies],axis=1)
weather_temp2.head(12)
#  Spliting X (Independent Variables) and Y(predicting variable)

Y = weather_temp["Apparent Temperature (C)"]
X = weather_temp2
# making Two dimensional array
#X = X.reshape(-1,1)
X
# Normalizing the X variables 
# check the x values should be between (-3  3)
#X.dropna()
x = (X >=0)
x.sum()
# prints the zero humidity
print(x.sum(),X.shape)
non_zero_humidity = X >= 0
print(non_zero_humidity.sum())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
X_train.shape, y_train.shape
#print(X_train.head(10))
y_train.head(10)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
X_train.shape,y_train.shape
plt.scatter(X_train["Humidity"], y_train)
#plt.plot(X_train["Humidity"], model.predict(X_train["Humidity"]), color='black')
plt.title("Temperature v/s humidity")
plt.xlabel("Humidity")
plt.ylabel("temperature")
pred = model.predict(X_test)
pred
y_predict = model.predict(X_test)
y_pred = y_predict.reshape(1,-1)
y_pred
y_test.head()
plt.scatter(X_test["Humidity"],y_predict,color='red')
#plt.plot(X_test["Humidity"],y_predict)
plt.title("Temperature v/s humidity")
plt.xlabel("Humidity")
plt.ylabel("temperature")
plt.scatter(X_test["Humidity"],y_test,color="green")
#plt.plot(X_test,y_test)
#X_test.shape
plt.title("Temperature v/s humidity")
plt.xlabel("Humidity")
plt.ylabel("temperature")
test = pd.DataFrame(y_test)
test["y_predict"]= 0
test["y_predict"] = y_predict
test.head(20)
weather.iloc[3703]
from sklearn.metrics import explained_variance_score
variance = explained_variance_score(y_test,y_predict, multioutput='uniform_average')
variance