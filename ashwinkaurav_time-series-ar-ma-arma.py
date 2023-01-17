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
data=pd.read_csv("../input/daily-temperature-of-major-cities/city_temperature.csv")

data.head()
#Taking out only Delhi data

delhi=data[data["City"]=="Delhi"]

delhi.reset_index(inplace=True)

delhi.drop('index',axis=1,inplace=True)

delhi.describe()           
import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))

plt.plot(delhi["AvgTemperature"])

plt.ylabel("Temperature",fontsize=20)
from sklearn.impute import SimpleImputer

imputer=SimpleImputer()

delhi["AvgTemperature"].replace(-99,np.nan,inplace=True)#Replacing wrong entries with nan 

delhi["AvgTemperature"]=pd.DataFrame(imputer.fit_transform(delhi.loc[:,"AvgTemperature":]))
print(min(delhi["AvgTemperature"]))

years=delhi["Year"].unique()

years
#Defining training and testing data

training_set=delhi[delhi["Year"]<=2015]

test_set=delhi[delhi["Year"]>2015]
#Mean of the temperatures

delhi.iloc[:,-1].mean()
plt.figure(figsize=(15,7))

plt.plot(delhi.iloc[:,-1])

plt.xlabel("Time Series",fontsize=20)

plt.ylabel("Temperature",fontsize=20)

#making a list of values to be plotted on y axis

y_values=[x for x in range(50,101,10)]

y_values.extend([delhi.iloc[:,-1].min(),delhi.iloc[:,-1].max(),delhi.iloc[:,-1].mean()])

plt.yticks(y_values)

plt.axhline(y=delhi.iloc[:,-1].mean(), color='r', linestyle='--',label="Mean")

plt.legend(loc=1)

plt.axhline(y=delhi.iloc[:,-1].max(), color='g', linestyle=':')

plt.axhline(y=delhi.iloc[:,-1].min(), color='g', linestyle=':')
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(delhi["AvgTemperature"],lags=365)

#plt.show()
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(delhi["AvgTemperature"],lags=50)
from statsmodels.tsa.ar_model import AutoReg

model_AR=AutoReg(training_set["AvgTemperature"],lags=365)

model_fit_AR=model_AR.fit()

predictions_AR = model_fit_AR.predict(training_set.shape[0],training_set.shape[0]+test_set.shape[0]-1)


import seaborn as sns

plt.figure(figsize=(15,5))

plt.ylabel("Temperature",fontsize=20)

plt.plot(test_set["AvgTemperature"],label="Original Data")

plt.plot(predictions_AR,label="Predicted Data")

plt.legend()
from sklearn.metrics import mean_squared_error

mse=mean_squared_error(predictions_AR,test_set["AvgTemperature"])

mse
from statsmodels.tsa.ar_model import AutoReg

model_AR2=AutoReg(training_set["AvgTemperature"],lags=[365])

model_fit_AR2=model_AR2.fit()

predictions_AR2= model_fit_AR2.predict(training_set.shape[0],training_set.shape[0]+test_set.shape[0]-1)
plt.figure(figsize=(15,5))

plt.ylabel("Temperature",fontsize=20)

plt.plot(test_set["AvgTemperature"],label="Original Data")

plt.plot(predictions_AR2,label="Predicted Data")

plt.legend()
mse=mean_squared_error(predictions_AR2,test_set["AvgTemperature"])

mse
from statsmodels.tsa.arima_model import ARMA

model_MA=ARMA(training_set["AvgTemperature"],order=(0,10))

model_fit_MA=model_MA.fit()

predictions_MA=model_fit_MA.predict(test_set.index[0],test_set.index[-1])
plt.figure(figsize=(15,5))

plt.ylabel("Temperature",fontsize=20)

plt.plot(test_set["AvgTemperature"],label="Original Data")

plt.plot(predictions_MA,label="Predictions")

plt.legend()
mse=mean_squared_error(predictions_MA,test_set["AvgTemperature"])

mse
model_ARMA=ARMA(training_set["AvgTemperature"],order=(5,10))

model_fit_ARMA=model_ARMA.fit()

predictions_ARMA=model_fit_ARMA.predict(test_set.index[0],test_set.index[-1])
plt.figure(figsize=(15,5))

plt.ylabel("Temperature",fontsize=20)

plt.plot(test_set["AvgTemperature"],label="Original Data")

plt.plot(predictions_ARMA,label="Predictions")

plt.legend()
mse=mean_squared_error(predictions_ARMA,test_set["AvgTemperature"])

mse