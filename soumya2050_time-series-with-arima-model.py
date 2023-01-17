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
# Import Libreary
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
%matplotlib inline
# Load the Data Set
testdata = pd.read_csv("/kaggle/input/predice-el-futuro/test_csv.csv")
traindata = pd.read_csv("/kaggle/input/predice-el-futuro/train_csv.csv")
# Check the data
traindata.info()
#Design the regex for date extraction 
#2019-03-19 00:00:10 ----Year(4digit)-Month(2digit)-Day(2digit) hour(2digit):min(2digit):second(2digit)
##                       %Y-%m-%d %H:%M:%S              

from pandas import datetime
def parserToTimeDatatype(time):
    return datetime.strptime(time,"%Y-%m-%d %H:%M:%S")
# Convert The time to Date Time format

finalData = pd.read_csv('/kaggle/input/predice-el-futuro/train_csv.csv',
                       parse_dates = [1], #Specify which column index has time info
                       index_col = 1, #Specify which column index as Index column
                       date_parser=parserToTimeDatatype) #Custom Parser
# Check the data after converting
finalData.info()
# Drop the ID column
finalData = finalData.iloc[:,[1]]
finalData
# Lets Plot the Data
plt.plot(finalData)
#Check whether the data is a stationary data or not

modifiedDF = finalData.diff(periods=1)
modifiedDF.dropna(inplace=True)
modifiedDF.head()
plt.plot(modifiedDF)
#Autocorrelation Plot --- To understand the behavior of data

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(modifiedDF)

#Out of 100% data, atleast 70% data must follow alternate pattern
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(modifiedDF)
#Create Train Test Split

features = modifiedDF.values
train = features[0:64]
test = features[64:]
#ARIMA Model (Moving Average)

from statsmodels.tsa.arima_model import ARIMA

#p - period
#d - Integral wrt period
#q - Moving Average Period 

p = 2
d = 1
q = 1

modelARIMA = ARIMA(train, order=(p,d,q))
finalARIMAI = modelARIMA.fit()
# Checking the model Summary
print(finalARIMAI.summary())
# # check the error score
finalARIMAI.aic
pred1 = finalARIMAI.forecast(steps=10)[0]
plt.plot(test)
plt.plot(pred1)
