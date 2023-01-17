#importing necessary libraries



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

import plotly.express as px

import datetime

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading the data

data = pd.read_csv('/kaggle/input/datasetucimlairquality/AirQualityUCI.csv')
data_1= pd.read_csv('/kaggle/input/datasetucimlairquality/AirQualityUCI.csv')
data.head()
#the number and type of data we have

data.info()
#to check for null values



data.isnull().any()
#setting time index



data.set_index("Date", inplace=True)

data.index = pd.to_datetime(data.index)

type(data.index)
data.head()
#converting the time to number value

#19.00.00 = 19 = 19 hours

#20.00.00 = 20 = 20 hours 

#and so on...

data['Time'] = pd.to_datetime(data['Time']).dt.hour
data.head()
#we have converted the 18.00.00 so 18 implying 6 pm and so on...

type(data["Time"][0])
#this column seems to have wrong or misplaced data, hence we are removing it



data.drop('NMHC_GT', axis=1, inplace=True)
#there seems to be lot of -200 for some reason, this is most likely an error, so we remove those values



data.replace(to_replace= -200, value= np.NaN, inplace= True)
#method to fill empty spaces

def VALUE_CORRECTION(col):

    data[col] = data.groupby('Date')[col].transform(lambda x: x.fillna(x.mean()))
#applying the value correction method in all the numeric columns

col_list = data.columns[1:13]



for i in col_list:

    VALUE_CORRECTION(i)
data.info()
#the parameter "ffill" stands for 'forward fill' and will propagate last valid observation forward



data.fillna(method='ffill', inplace= True)
data.info()
#scatter plot 



fig = px.scatter(data, x="Time",y="CO_GT", 

              title="True hourly averaged concentration CO in mg/m^3 at various times of Day")

fig.show()
plt.title("True hourly averaged concentration CO in mg/m^3 Distribution")

sns.distplot(data['CO_GT'])
fig = px.scatter(data, x="Time",y="PT08_S1_CO", 

              title="PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted) at various times of Day")

fig.show()
plt.title("PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted) Distribution")

sns.distplot(data['PT08_S1_CO'])
fig = px.scatter(data, x="Time",y="PT08_S2_NMHC", 

              title="True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 at various times of Day")

fig.show()
plt.title("True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 Distribution")

sns.distplot(data['PT08_S2_NMHC'])
fig = px.scatter(data, x="Time",y="C6H6_GT", 

              title="True hourly averaged Benzene concentration in microg/m^3 at various times of Day")

fig.show()
plt.title("True hourly averaged Benzene concentration in microg/m^3 Distribution")

sns.distplot(data['C6H6_GT'])
data.head()
#Temperature and Relative Humidity
plt.xlabel("Temperature in Degree Celsius")

plt.ylabel('Relative Humidity')

plt.xlim(-5,10)

plt.title("Relative Humidity vs Temperature")

plt.scatter(data['T'],data["RH"],marker=".")
plt.xlabel("Temperature in Degree Celsius")

plt.ylabel('Relative Humidity')

plt.xlim(10,25)

plt.title("Relative Humidity vs Temperature")

plt.scatter(data['T'],data["RH"],marker=".")
plt.xlabel("Temperature in Degree Celsius")

plt.ylabel('Relative Humidity')

plt.xlim(25,30)

plt.title("Relative Humidity vs Temperature")

plt.scatter(data['T'],data["RH"],marker=".")
plt.xlabel("Temperature in Degree Celsius")

plt.ylabel('Relative Humidity')

plt.xlim(30,50)

plt.title("Relative Humidity vs Temperature")

plt.scatter(data['T'],data["RH"],marker=".")
plt.xlabel("Temperature in Degree Celsius")

plt.ylabel('Relative Humidity')

plt.title("Relative Humidity vs Temperature-Full")

plt.scatter(data['T'],data["RH"],marker=".")
#Relative Humidity vs Temperature

a=sns.jointplot("T", "RH", data=data,kind= "kde").set_axis_labels("Temperature In Degree Celsius", "Relative Humidity")
#Relative Humidity vs Temperature

a=sns.jointplot("T", "RH", data= data,kind="hex").set_axis_labels("Temperature In Degree Celsius", "Relative Humidity")
#Relative Humidity vs Temperature

a=sns.jointplot("T", "RH", data= data,kind="reg").set_axis_labels("Temperature In Degree Celsius", "Relative Humidity")
data.head()
#taking only the Temperature and Relative Humidity



data_subset1=data[["T","RH"]]

plt.figure(figsize=(16, 6))

sns.lineplot(data=data_subset1, linewidth=1)
#Taking only the CO and C6H6 values



data_subset2=data[["CO_GT","C6H6_GT"]]

plt.figure(figsize=(16, 6))

sns.lineplot(data=data_subset2, linewidth=1)
plt.figure(figsize=(16, 6))

sns.lineplot(data=data['Nox_GT'], linewidth=1)
plt.figure(figsize=(16, 6))

sns.lineplot(data=data['PT08_S3_Nox'], linewidth=1)
#Thank You