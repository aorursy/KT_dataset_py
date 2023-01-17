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
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from datetime import timedelta

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,r2_score

import statsmodels.api as sm

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

from fbprophet import Prophet

from sklearn.preprocessing import PolynomialFeatures

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head()
df.info()
df.dtypes
df.describe()
df.isnull()
df.replace('NaN', 'False')
#Converting "Observation Date" into Datetime format

df["ObservationDate"]=pd.to_datetime(df["ObservationDate"])
datewise = df.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise.head()
countrywise = df.groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

countrywise.head()
print("Totol number of countries with Disease Spread: ",len(df["Country/Region"].unique()))

print("Total number of Confirmed Cases",datewise["Confirmed"].iloc[-1])

print("Total number of Recovered Cases",datewise["Recovered"].iloc[-1])

print("Total number of Deaths Cases",datewise["Deaths"].iloc[-1])
plt.figure(figsize=(12,6))

plt.plot(datewise["Confirmed"],marker="o",label="Confirmed Cases")

plt.plot(datewise["Recovered"],marker="*",label="Recovered Cases")

plt.plot(datewise["Deaths"],marker="^",label="Death Cases")

plt.ylabel("Number of Patients")

plt.xlabel("Timestamp")

plt.xticks(rotation=90)

plt.title("Growth of different Types of Cases over Time")

plt.legend()
#Calculating the Mortality Rate and Recovery Rate

datewise["Mortality Rate"]=(datewise["Deaths"]/datewise["Confirmed"])*100

datewise["Recovery Rate"]=(datewise["Recovered"]/datewise["Confirmed"])*100



#Plotting Mortality and Recovery Rate 

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,6))

ax1.plot(datewise["Mortality Rate"],label='Mortality Rate')

ax1.axhline(datewise["Mortality Rate"].mean(),linestyle='--',color='black',label="Mean Mortality Rate")

ax1.set_ylabel("Number of Patients")

ax1.set_xlabel("Timestamp")

ax1.legend()

for tick in ax1.get_xticklabels():

    tick.set_rotation(90)

ax2.plot(datewise["Recovery Rate"],label="Recovery Rate")

ax2.axhline(datewise["Recovery Rate"].mean(),linestyle='--',color='black',label="Mean Recovery Rate")

ax2.set_ylabel("Number of Patients")

ax2.set_xlabel("Timestamp")

ax2.legend()

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)
#Calculating countrywise Moratality and Recovery Rate

countrywise=df[df["ObservationDate"]==df["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed"],ascending=False)

countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100

countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(27,8))

countrywise_plot_mortal=countrywise[countrywise["Confirmed"]>50].sort_values(["Mortality"],ascending=False).head(25)

sns.barplot(x=countrywise_plot_mortal["Mortality"],y=countrywise_plot_mortal.index,ax=ax1)

ax1.set_title("Top 25 Countries according Mortatlity Rate")

ax1.set_xlabel("Mortality (in Percentage)")

countrywise_plot_recover=countrywise[countrywise["Confirmed"]>100].sort_values(["Recovery"],ascending=False).head(25)

sns.barplot(x=countrywise_plot_recover["Recovery"],y=countrywise_plot_recover.index, ax=ax2)

ax2.set_title("Top 25 Countries according Recovery Rate")

ax2.set_xlabel("Recovery (in Percentage)")
deaths=countrywise[(countrywise["Deaths"]>50)&(countrywise["Recovered"]>0)]

deaths[deaths["Confirmed"]>0].sort_values(["Confirmed"],ascending=False)
no_deaths=countrywise[(countrywise["Confirmed"]>100)&(countrywise["Deaths"]==0)]

no_deaths[no_deaths["Recovery"]>0].sort_values(["Recovery"],ascending=False)
mostRecov=countrywise[(countrywise["Recovery"]>=20)&(countrywise["Deaths"]>0)]

mostRecov[mostRecov["Confirmed"]>0].sort_values(["Confirmed"],ascending=False)
Thai_data=df[df["Country/Region"]=="Thailand"]

datewise_Thai=Thai_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_Thai.iloc[-1])
plt.figure(figsize=(10,6))

plt.plot(datewise_Thai["Confirmed"],marker='o',label="Confirmed Cases")

plt.plot(datewise_Thai["Recovered"],marker='*',label="Recovered Cases")

plt.plot(datewise_Thai["Deaths"],marker='^',label="Death Cases")

plt.ylabel("Number of Patients")

plt.xlabel("Date")

plt.legend()

plt.title("Growth Rate Plot for different Types of cases in Thailand")

plt.xticks(rotation=90)
Thai_increase_confirm=[]

Thai_increase_recover=[]

Thai_increase_deaths=[]

for i in range(datewise_Thai.shape[0]-1):

    Thai_increase_confirm.append(((datewise_Thai["Confirmed"].iloc[i+1]-datewise_Thai["Confirmed"].iloc[i])/datewise_Thai["Confirmed"].iloc[i]))

    Thai_increase_recover.append(((datewise_Thai["Recovered"].iloc[i+1]-datewise_Thai["Recovered"].iloc[i])/datewise_Thai["Recovered"].iloc[i]))

    Thai_increase_deaths.append(((datewise_Thai["Deaths"].iloc[i+1]-datewise_Thai["Deaths"].iloc[i])/datewise_Thai["Deaths"].iloc[i]))

Thai_increase_confirm.insert(0,0)

Thai_increase_recover.insert(0,0)

Thai_increase_deaths.insert(0,0)



plt.figure(figsize=(10,6))

plt.plot(datewise_Thai.index,Thai_increase_confirm,label="Growth Rate of Confirmed Cases",marker='o')

plt.plot(datewise_Thai.index,Thai_increase_recover,label="Growth Rate of Recovered Cases",marker='*')

plt.plot(datewise_Thai.index,Thai_increase_deaths,label="Growth Rate of Death Cases",marker='^')

plt.xticks(rotation=90)

plt.title("Datewise Growth Rate of different Types of Cases")

plt.legend()
# Group by the segment label and calculate average column values

datewise_Thai_averages = datewise_Thai.groupby(['Comfirmed']).mean().round(0)



# Print the average column values per each segment

print(datewise_Thai_averages.describe())
# Create a heatmap on the average column values per each segment

sns.heatmap(datewise_Thai_averages,cmap='RdYlGn')



# Display the chart

plt.show()
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, classification_report

from sklearn.metrics import accuracy_score, f1_score
# Import KNeighborsClassifier from sklearn.neighbors

from sklearn.neighbors import KNeighborsClassifier



# Create arrays for the features and the response variable

y = datewise['Deaths'].values

X = datewise.drop('Recovered', axis=1).values



# Create a k-NN classifier with 6 neighbors

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X,y)
# Predict the labels for the training data X

y_pred = knn.predict(X)
print(classification_report(y_pred, y))