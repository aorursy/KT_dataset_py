# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt



%matplotlib inline

plt.style.use('seaborn-whitegrid')



import warnings



warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
austin_weather = pd.read_csv('/kaggle/input/austin-weather/austin_weather.csv', parse_dates = ['Date'])



plt.rcParams['figure.figsize'] = [14, 10]



plt.xlabel('Days')

plt.ylabel('Temperature AVG (°F)')

plt.scatter(austin_weather.Date, austin_weather.TempAvgF, marker='.', color='black')

plt.show()
temperatures = list(austin_weather.TempAvgF) 



x_train = []

for y in austin_weather.index:

    x_train.append([y])



reg = linear_model.Ridge(alpha=0)

reg.fit(x_train, temperatures) 

prediction = reg.predict(x_train)



plt.xlabel('Days')

plt.ylabel('Temperature AVG (°F)')

plt.scatter(austin_weather.Date, austin_weather.TempAvgF, marker='.', color='black')

plt.plot(austin_weather.Date, prediction, color='green', linewidth='1')

plt.show()

reg2 = linear_model.Ridge(alpha=50000000)

reg3 = linear_model.Ridge(alpha=500000000)



reg2.fit(x_train, temperatures)

prediction2 = reg2.predict(x_train)



reg3.fit(x_train, temperatures) 

prediction3 = reg3.predict(x_train)



plt.xlabel('Days')

plt.ylabel('Temperature AVG (°F)')

plt.scatter(austin_weather.Date, austin_weather.TempAvgF, marker='.', color='black')

plt.plot(austin_weather.Date, prediction, color='green', linewidth='1')

plt.plot(austin_weather.Date, prediction2, color='blue', linewidth='1')

plt.plot(austin_weather.Date, prediction3, color='red', linewidth='1')

plt.legend(['alpha = 0','alpha = 50.000.000','alpha = 500.000.000'], numpoints=1)

plt.show()
alphas = [0, 50000000, 500000000]

predictions = []

for i in range(0, 3):

    model = make_pipeline(PolynomialFeatures(2), linear_model.Ridge(alpha=alphas[i]))

    model.fit(x_train, temperatures)

    predictions.append(model.predict(x_train))

    

plt.xlabel('Days')

plt.ylabel('Temperature AVG (°F)')

plt.scatter(austin_weather.Date, austin_weather.TempAvgF, marker='.', color='black')

plt.plot(austin_weather.Date, predictions[0], color='green', linewidth='1')

plt.plot(austin_weather.Date, predictions[1], color='blue', linewidth='1')

plt.plot(austin_weather.Date, predictions[2], color='red', linewidth='1')

plt.legend(['alpha = 0','alpha = 50.000.000','alpha = 500.000.000'], numpoints=1)

plt.show()
alphas = [0, 50000000, 500000000]

degrees = [1, 2, 4]

predictions = []

for i in range(0, 3):

    model = make_pipeline(PolynomialFeatures(degrees[i]), linear_model.Ridge(alpha=alphas[i]))

    model.fit(x_train, temperatures)

    predictions.append(model.predict(x_train))

    

plt.xlabel('Days')

plt.ylabel('Temperature AVG (°F)')

plt.scatter(austin_weather.Date, austin_weather.TempAvgF, marker='.', color='black')

plt.plot(austin_weather.Date, predictions[0], color='green', linewidth='1')

plt.plot(austin_weather.Date, predictions[1], color='blue', linewidth='1')

plt.plot(austin_weather.Date, predictions[2], color='red', linewidth='1')

plt.legend(['degree = 1', 'degree  = 2', 'degree = 4'], numpoints=1)

plt.show()