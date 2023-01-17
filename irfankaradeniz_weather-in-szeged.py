# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
weatherData = pd.read_csv('../input/szeged-weather/weatherHistory.csv',encoding='latin1')
for i in weatherData.columns:

    print(i)
apparenttemp = weatherData[['Apparent Temperature (C)']]

temperature = weatherData[['Temperature (C)']]

humidity = weatherData[['Humidity']]



humidity.isnull().sum()
plt.scatter(humidity, temperature, edgecolors='r')

plt.xlabel('humidity')

plt.ylabel('temperature')

plt.show()
weatherData['Temperature (C)'].corr(weatherData['Humidity'])
plt.scatter(humidity, apparenttemp, edgecolors='r')

plt.xlabel('humidity')

plt.ylabel('apparent temperature')

plt.show()
weatherData['Apparent Temperature (C)'].corr(weatherData['Humidity'])
from sklearn.model_selection  import train_test_split

x_train, x_test, y_train, y_test = train_test_split(humidity,apparenttemp,test_size=0.33,random_state=0)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)
prediction = lr.predict(x_test)
plt.plot(x_train,y_train)

plt.plot(x_test,prediction)

plt.show()