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

from sklearn.metrics import mean_squared_error,r2_score, classification_report

import statsmodels.api as smA

import random

import matplotlib.colors as mcolors

from sklearn.tree import DecisionTreeClassifier
covid=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid.head()
print("Size/Shape of the dataset: ",covid.shape)

print("Checking for null values:\n",covid.isnull().sum())

print("Checking Data-type of each column:\n",covid.dtypes)

covid.info()
covid.dtypes
covid.describe()
#Converting "Observation Date" into Datetime format

covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])
datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise.head()
print("Totol number of countries with Disease Spread: ",len(covid["Country/Region"].unique()))

print("Total number of Confirmed Cases",datewise["Confirmed"].iloc[-1])

print("Total number of Recovered Cases",datewise["Recovered"].iloc[-1])

print("Total number of Deaths Cases",datewise["Deaths"].iloc[-1])
#Distribution plot of confirmed cases around the world 

sns.kdeplot(datewise["Confirmed"])

plt.title("Density Distribution Plot for Confirmed Cases")
sns.kdeplot(datewise["Deaths"])

plt.title("Density Distribution Plot for Death Cases")
sns.kdeplot(datewise["Recovered"])

plt.title("Density Distribution Plot for Recovered Cases")
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

countrywise=covid[covid["ObservationDate"]==covid["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed"],ascending=False)

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
no_deaths=countrywise[(countrywise["Confirmed"]>100)&(countrywise["Deaths"]==0)]

no_deaths[no_deaths["Recovery"]>0].sort_values(["Recovery"],ascending=False)
china_data=covid[covid["Country/Region"]=="Mainland China"]

Italy_data=covid[covid["Country/Region"]=="Italy"]

rest_of_world=covid[(covid["Country/Region"]!="Mainland China")&(covid["Country/Region"]!="Italy")]



datewise_china=china_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_Italy=Italy_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise_restofworld=rest_of_world.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(12,30))

ax1.plot(datewise_china["Confirmed"],label="Confirmed Cases of Mainland China",marker='o')

ax1.plot(datewise_Italy["Confirmed"],label="Confirmed Cases of Italy",marker='*')

ax1.plot(datewise_restofworld["Confirmed"],label="Confirmed Cases of Rest of the World",marker='o')

ax1.set_title("Confirmed Cases Plot")

ax1.set_ylabel("Number of Patients")

ax1.set_xlabel("Timestamp")

ax1.legend()

for tick in ax1.get_xticklabels():

    tick.set_rotation(45)

ax2.plot(datewise_china["Recovered"],label="Recovered Cases of Mainland China",marker='o')

ax2.plot(datewise_Italy["Recovered"],label="Recovered Cases of Italy",marker='*')

ax2.plot(datewise_restofworld["Recovered"],label="Recovered Cases of Rest of the World",marker='^')

ax2.set_title("Recovered Cases Plot")

ax2.set_ylabel("Number of Patients")

ax2.set_xlabel("Timestamp")

ax2.legend()

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)

ax3.plot(datewise_china["Deaths"],label='Death Cases of Mainland China',marker='o')

ax3.plot(datewise_Italy["Deaths"],label='Death Cases of Italy',marker='*')

ax3.plot(datewise_restofworld["Deaths"],label="Deaths Cases of Rest of the World",marker='^')

ax3.set_title("Death Cases Plot")

ax3.set_ylabel("Number of Patients")

ax3.set_xlabel("Timestamp")

ax3.legend()

for tick in ax3.get_xticklabels():

    tick.set_rotation(90)
Thai_data=covid[covid["Country/Region"]=="Thailand"]

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

datewise_Thai_averages = datewise_Thai.groupby(['Confirmed']).mean().round(0)



# Print the average column values per each segment

print(datewise_Thai_averages.describe())
# Create a heatmap on the average column values per each segment

sns.heatmap(datewise_Thai_averages.T, cmap='YlGnBu')



# Display the chart

plt.show()


from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

x = datewise.drop(columns=['Deaths'],axis=1)

y = datewise['Deaths']

# แบ่ง X_train, X_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
mytree = DecisionTreeClassifier()
mytree.fit(x_test, y_test)
predictions = mytree.predict(x_test)
print(classification_report(predictions, y_test))