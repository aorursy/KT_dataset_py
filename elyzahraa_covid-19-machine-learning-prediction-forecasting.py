# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import seaborn as sns

import datetime as dt

from datetime import timedelta

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import RandomizedSearchCV, train_test_split 

from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.api import Holt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



covid = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid.head(10)
covid.tail()
covid.shape
rng = np.random.RandomState(42)

ser = pd.Series(rng.rand(5))

ser
print("Size/Shape of dataset", covid.shape)

print("===================================")

print("checking for null values", covid.isnull().sum())

print("===================================")

print("checking data-type", covid.dtypes)
covid.drop(["SNo"],1, inplace=True)

covid.isnull().sum()
covid["ObservationDate"] = pd.to_datetime(covid["ObservationDate"])
covid["ObservationDate"] 
datewise = covid.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
print("Basic Information")

print("Total number of Confirmed cases around the world :", datewise["Confirmed"].iloc[-1])

print("Total number of Recovered cases around the world :", datewise["Recovered"].iloc[-1])

print("Total number of Deaths cases around the world    :", datewise["Deaths"].iloc[-1])

print("Total number of Active cases around the world    :", (datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))

print("Total number of Closed cases around the world    :", (datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1]))
plt.figure(figsize=(30,10))

sns.barplot(x=datewise.index.date,y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])

plt.title("Covid-19 Active Cases")

plt.xticks(rotation=90)
plt.figure(figsize=(30,10))

sns.barplot(x=datewise.index.date,y=datewise["Recovered"]+datewise["Deaths"])

plt.title("Covid-19 Closed Cases")

plt.xticks(rotation=90)
datewise["WeekofYear"]= datewise.index.weekofyear

week_num = []

weekwise_confirmed = []

weekwise_recovered = []

weekwise_deaths = []

w = 1

for i in list(datewise["WeekofYear"].unique()):

    weekwise_confirmed.append(datewise[datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])

    weekwise_recovered.append(datewise[datewise["WeekofYear"]==i]["Recovered"].iloc[-1])

    weekwise_deaths.append(datewise[datewise["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num.append(w)

    w=w+1

    

plt.figure(figsize=(8,5))

plt.plot(week_num,weekwise_confirmed,linewidth=3)

plt.plot(week_num,weekwise_recovered,linewidth=3)

plt.plot(week_num,weekwise_deaths,linewidth=3)

plt.xlabel("Week of Number")

plt.ylabel("Number of cases")

plt.title("Weekly Progress of different type of cases")

plt.show()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(17,4))

sns.barplot(x= week_num,y=pd.Series(weekwise_confirmed).diff().fillna(0),ax=ax1)

sns.barplot(x= week_num,y=pd.Series(weekwise_deaths).diff().fillna(0),ax=ax2)

ax1.set_xlabel("Week Number")

ax2.set_xlabel("Week Number")

ax1.set_ylabel("Number of Confirmed cases")

ax2.set_ylabel("Number of Deaths cases")

ax1.set_title("Weekly increase in number of Confirmed cases")

ax2.set_title("Weekly increase in number of Deaths cases")

plt.show()
print("Average increase in number of Confirmed cases everyday:",np.round(datewise["Confirmed"].diff().fillna(0).mean()))

print("Average increase in number of Recovered cases everyday:",np.round(datewise["Recovered"].diff().fillna(0).mean()))

print("Average increase in number of Deaths cases everyday:",np.round(datewise["Deaths"].diff().fillna(0).mean()))



plt.figure(figsize=(10,6))

plt.plot(datewise["Confirmed"].diff().fillna(0),label="daily increase in confirmed cases",linewidth=3)

plt.plot(datewise["Recovered"].diff().fillna(0),label="daily increase in recovered cases",linewidth=3)

plt.plot(datewise["Deaths"].diff().fillna(0),label="daily increase in deaths cases",linewidth=3)



plt.xlabel("Timestamp")

plt.ylabel("Daily increase")

plt.title("Daily increase")

plt.legend()

plt.xticks(rotation=90)
#Country wise analysis

#Calculating Country wise Mortality rate



countrywise=covid[covid["ObservationDate"]==covid["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed"],ascending=False)

countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Confirmed"])*100

countrywise["Recovery"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,10))

top_20confirm = countrywise.sort_values(["Confirmed"],ascending=False).head(20)

top_20deaths = countrywise.sort_values(["Deaths"],ascending=False).head(20)



sns.barplot(x=top_20confirm["Confirmed"],y=top_20confirm.index,ax=ax1)

ax1.set_title("Top 20 Countries as per number of confirmed cases")

sns.barplot(x=top_20deaths["Deaths"],y=top_20confirm.index,ax=ax2)

ax2.set_title("Top 20 Countries as per number of deaths cases")

#Data Analysis for Indonesian



indonesia_data = covid[covid["Country/Region"]=="Indonesia"]

datewise_indonesia = indonesia_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_indonesia.iloc[-1])

print("=====================================")

print("Total Active Cases", datewise_indonesia["Confirmed"].iloc[-1]-datewise_indonesia["Recovered"].iloc[-1]-datewise_indonesia["Deaths"].iloc[-1])

print("Total Closed Cases", datewise_indonesia["Recovered"].iloc[-1]+datewise_indonesia["Deaths"].iloc[-1])
#Data Analysis for US



US_data = covid[covid["Country/Region"]=="US"]

datewise_US = US_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

print(datewise_US.iloc[-1])

print("=====================================")

print("Total Active Cases", datewise_US["Confirmed"].iloc[-1]-datewise_US["Recovered"].iloc[-1]-datewise_US["Deaths"].iloc[-1])

print("Total Closed Cases", datewise_US["Recovered"].iloc[-1]+datewise_US["Deaths"].iloc[-1])
#WEEKLY DATA ON INDONESIAN



datewise_indonesia["WeekofYear"] = datewise_indonesia.index.weekofyear

week_num_ina = []

weekwise_confirmed_ina = []

weekwise_recovered_ina = []

weekwise_deaths_ina = []

w = 1

for i in list(datewise_indonesia["WeekofYear"].unique()):

    weekwise_confirmed_ina.append(datewise_indonesia[datewise_indonesia["WeekofYear"]==i]["Confirmed"].iloc[-1])

    weekwise_recovered_ina.append(datewise_indonesia[datewise_indonesia["WeekofYear"]==i]["Recovered"].iloc[-1])

    weekwise_deaths_ina.append(datewise_indonesia[datewise_indonesia["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_ina.append(w)

    w=w+1

    

plt.figure(figsize=(8,5))

plt.plot(week_num_ina,weekwise_confirmed_ina,linewidth=3)

plt.plot(week_num_ina,weekwise_recovered_ina,linewidth=3)

plt.plot(week_num_ina,weekwise_deaths_ina,linewidth=3)

plt.xlabel("Week of Number in Indonesian")

plt.ylabel("Number of cases in Indonesian")

plt.title("Weekly Progress of different type of cases in Indonesian")

plt.show()
#WEEKLY DATA ON US



datewise_US["WeekofYear"] = datewise_US.index.weekofyear

week_num_us = []

weekwise_confirmed_us = []

weekwise_recovered_us = []

weekwise_deaths_us = []

w = 1

for i in list(datewise_US["WeekofYear"].unique()):

    weekwise_confirmed_us.append(datewise_US[datewise_US["WeekofYear"]==i]["Confirmed"].iloc[-1])

    weekwise_recovered_us.append(datewise_US[datewise_US["WeekofYear"]==i]["Recovered"].iloc[-1])

    weekwise_deaths_us.append(datewise_US[datewise_US["WeekofYear"]==i]["Deaths"].iloc[-1])

    week_num_us.append(w)

    w=w+1

    

plt.figure(figsize=(8,5))

plt.plot(week_num_us,weekwise_confirmed_us,linewidth=3)

plt.plot(week_num_us,weekwise_recovered_us,linewidth=3)

plt.plot(week_num_us,weekwise_deaths_us,linewidth=3)

plt.xlabel("Week of Number in US")

plt.ylabel("Number of cases in US")

plt.title("Weekly Progress of different type of cases in US")

plt.show()
max_ina = datewise_indonesia["Confirmed"].max()

china_data = covid[covid["Country/Region"]=="Mainland China"]

southKor_data = covid[covid["Country/Region"]=="South Korea"]

germany_data = covid[covid["Country/Region"]=="Germany"]

US_data = covid[covid["Country/Region"]=="US"]

italy_data = covid[covid["Country/Region"]=="Italy"]

spain_data = covid[covid["Country/Region"]=="Spain"]



china = china_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

southKorea = southKor_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

germany = germany_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

US = US_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

italy = italy_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

spain = spain_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})



print("It took", datewise_indonesia[datewise_indonesia["Confirmed"]>0].shape[0],"days in Indonesian to reach",max_ina,"Confirmed Cases")

print("It took", china[(china["Confirmed"]>0)&(china["Confirmed"]<=max_ina)].shape[0],"days in  China to reach number of Confirmed Cases")

print("It took", southKorea[(southKorea["Confirmed"]>0)&(southKorea["Confirmed"]<=max_ina)].shape[0],"days in  South Korea to reach number of Confirmed Cases")

print("It took", germany[(germany["Confirmed"]>0)&(germany["Confirmed"]<=max_ina)].shape[0],"days in  Germany to reach number of Confirmed Cases")

print("It took", US[(US["Confirmed"]>0)&(US["Confirmed"]<=max_ina)].shape[0],"days in  US to reach number of Confirmed Cases")

print("It took", italy[(italy["Confirmed"]>0)&(italy["Confirmed"]<=max_ina)].shape[0],"days in  Italy to reach number of Confirmed Cases")

print("It took", spain[(spain["Confirmed"]>0)&(spain["Confirmed"]<=max_ina)].shape[0],"days in  Spain to reach number of Confirmed Cases")
datewise["Days Since"] = datewise.index-datewise.index[0]

datewise["Days Since"] = datewise["Days Since"].dt.days

train = datewise.iloc[:int(datewise.shape[0]*0.95)]

valid =  datewise.iloc[:int(datewise.shape[0]*0.95)]

model_scores=[]
reg = LinearRegression(normalize=True)

svm = SVR(C=1,degree=5,kernel='poly',epsilon=0.001)

reg.fit(np.array(train["Days Since"]).reshape(-1,1),np.array(train["Confirmed"]).reshape(-1,1))

svm.fit(np.array(train["Days Since"]).reshape(-1,1),np.array(train["Confirmed"]).reshape(-1,1))

prediction_valid_reg = reg.predict(np.array(valid["Days Since"]).reshape(-1,1))

prediction_valid_svm = svm.predict(np.array(valid["Days Since"]).reshape(-1,1))
new_date = []

new_predict_lr = []

new_predict_svm = []

for i in range(1,18):

    new_date.append(datewise.index[-1]+timedelta(days=i))

    new_predict_lr.append(reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])

    new_predict_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])

pd.set_option("display.float_format",lambda x: '%.f' % x)

model_predict = pd.DataFrame(zip(new_date,new_predict_lr,new_predict_svm),columns = ["Dates","LR","SVR"])

model_predict.head(10)
model_train = datewise.iloc[:int(datewise.shape[0]*0.95)]

valid = datewise.iloc[int(datewise.shape[0]*0.85):]
holt = Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=1.4,smoothing_slope=0.2)

y = valid.copy()

y["Holt"] = holt.forecast(len(valid))
holt_new_date=[]

holt_new_predict=[]



for i in range(1,18):

    holt_new_date.append(datewise.index[-1]+timedelta(days=i))

    holt_new_predict.append(holt.forecast((len(valid)+i)) [-1])



model_predict["Holts Linear Model Predictions"]=holt_new_predict

model_predict.head()