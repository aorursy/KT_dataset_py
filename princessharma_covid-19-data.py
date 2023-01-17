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
import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import datetime as dt

from datetime import timedelta

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,r2_score

import statsmodels.api as sm

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

#from fbprophet import Prophet

from sklearn.preprocessing import PolynomialFeatures

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

pd.set_option('display.float_format', lambda x: '%.6f' % x)
# to get web contents

import requests

# to parse json contents

import json

# to parse csv files

import csv



# for numerical operations

import numpy as np

# to store and analysis data in dataframes

import pandas as pd


#df = pd.DataFrame('https://api.covid19india.org/csv/latest/state_wise.csv')

df = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv')
df.head()
print("Size/Shape of the dataset: ",df.shape)

print("Checking for null values:\n",df.isnull().sum())

print("Checking Data-type of each column:\n",df.dtypes)
#Grouping different types of cases as per the date

datewise=df.groupby(["Last_Updated_Time"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise
print("Basic Information")

print("Total number of States/UT with Disease Spread: ",len(df["State"].unique()))

print("Total number of Confirmed Cases around the India: ",df["Confirmed"].iloc[0])

print("Total number of Recovered Cases around the India: ",df["Recovered"].iloc[0])

print("Total number of Deaths Cases around the India: ",df["Deaths"].iloc[0])



print("Total number of Active Cases in India: ",(df["Confirmed"].iloc[0]-df["Recovered"].iloc[0]-df["Deaths"].iloc[0]))

print("Total number of Closed Cases in India: ",df["Recovered"].iloc[0]+df["Deaths"].iloc[0])



plt.figure(figsize=(15,5))

sns.barplot(x=df["State"], y=df["Confirmed"]-df["Recovered"]-df["Deaths"])

plt.title("Distribution Plot for Active Cases Cases over State")

plt.xticks(rotation=90)
plt.figure(figsize=(12,6))

sns.barplot(x=df["State"], y=df["Recovered"]+df["Deaths"])

plt.title("Distribution Plot for Closed Cases Cases over Date")

plt.xticks(rotation=90)
plt.figure(figsize=(15,6))

plt.plot(df["Confirmed"].iloc[0:],marker="o",label="Confirmed Cases")

plt.plot(df["Recovered"].iloc[0:],marker="*",label="Recovered Cases")

plt.plot(df["Deaths"].iloc[0:],marker="^",label="Death Cases")

plt.ylabel("Number of Patients")

plt.xlabel("time")



plt.title("Growth of Types of Cases over Time")

plt.legend()
#Calculating the Mortality Rate and Recovery Rate

df["Mortality Rate"]=(df["Deaths"]/df["Confirmed"])*100

df["Recovery Rate"]=(df["Recovered"]/df["Confirmed"])*100

df["Active Cases"]=df["Confirmed"]-df["Recovered"]-df["Deaths"]

df["Closed Cases"]=df["Recovered"]+df["Deaths"]



#Plotting Mortality and Recovery Rate 

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,6))

ax1.plot(df["Mortality Rate"],label='Mortality Rate',linewidth=3)

ax1.axhline(df["Mortality Rate"].mean(),linestyle='--',color='black',label="Mean Mortality Rate")

ax1.set_ylabel("Mortality Rate")

ax1.set_xlabel("Timestamp")

ax1.legend()

for tick in ax1.get_xticklabels():

    tick.set_rotation(90)

ax2.plot(df["Recovery Rate"],label="Recovery Rate",linewidth=3)

ax2.axhline(df["Recovery Rate"].mean(),linestyle='--',color='black',label="Mean Recovery Rate")

ax2.set_ylabel("Recovery Rate")

ax2.set_xlabel("Timestamp")

ax2.legend()

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)

    

print("Average Mortality Rate",df["Mortality Rate"].mean())

print("Median Mortality Rate",df["Mortality Rate"].median())

print("Average Recovery Rate",df["Recovery Rate"].mean())

print("Median Recovery Rate",df["Recovery Rate"].median())