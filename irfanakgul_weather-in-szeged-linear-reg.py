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
data_all= pd.read_csv("/kaggle/input/szeged-weather/weatherHistory.csv")
data_all.rename(columns={"Apparent Temperature (C)": "app_temp", "Humidity": "humidity"}, inplace=True)   

app_temp = data_all.app_temp

humidity = data_all.humidity

data =pd.concat([app_temp,humidity],axis=1, ignore_index=True)

data.rename(columns={0: "app_temp", 1: "humidity"}, inplace=True)
data.head()
data.isna().sum()
data.corr()
data.info()
data.describe().T
data.count()
x=data[["humidity"]]

y=data["app_temp"]
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state= 0)
model = LinearRegression().fit(x_train,y_train)  # ornek modeli tanimliyoruz
y_pred = model.predict(x_test)

y_pred
from sklearn.metrics import mean_squared_error, r2_score

r2_score(y_test, y_pred)
model.predict([[0.89]])
import matplotlib.pyplot as plt

import seaborn as sns

aray = np.arange(len(y_test))

plt.plot(aray, y_pred, color="red" )  

plt.plot(aray, y_test, color="blue",alpha=0.5)



plt.show();
plt.plot(x_test, y_test,  color='black')

plt.plot(y_test, y_pred, color='blue', linewidth=3)



plt.xticks(())

plt.yticks(())



plt.show()