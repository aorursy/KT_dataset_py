import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

%matplotlib inline

import seaborn as sns

from scipy.integrate import odeint





cdr_logs = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", index_col="ObservationDate") ##dataset w/ EVERY country (CONFIRMED, DEATHS, RECOVERED)

cdr_logs["Last Update"] = pd.to_datetime(cdr_logs["Last Update"])#change date format to "datetime64"

cdr_logs["Province/State"] = cdr_logs["Province/State"].fillna("No data")#Change NaN states for "No data" string

cdr_logs_argentina = cdr_logs[cdr_logs["Country/Region"]=="Argentina"] ##dataset ARGENTINA (CONFIRMED, DEATHS, RECOVERED)

cdr_logs_singapore = cdr_logs[cdr_logs["Country/Region"]=="Singapore"] ##dataset SINGAPORE (CONFIRMED, DEATHS, RECOVERED)

cdr_logs_germany = cdr_logs[cdr_logs["Country/Region"]=="Germany"] ##dataset GERMANY (CONFIRMED, DEATHS, RECOVERED)

cdr_logs_japan = cdr_logs[cdr_logs["Country/Region"]=="Japan"] ##dataset JAPAN (CONFIRMED, DEATHS, RECOVERED)

cdr_logs_skorea = cdr_logs[cdr_logs["Country/Region"]=="South Korea"] ##dataset SOUTH KOREA (CONFIRMED, DEATHS, RECOVERED)

plt.figure(figsize=(40,20))

sns.lineplot(data = cdr_logs_argentina.loc[:,["Confirmed","Deaths","Recovered"]])
plt.figure(figsize=(40,20))

sns.lineplot(data = cdr_logs_singapore.loc[:,["Confirmed","Deaths","Recovered"]])
plt.figure(figsize=(40,20))

sns.lineplot(data = cdr_logs_germany.loc[:,["Confirmed","Deaths","Recovered"]])

plt.figure(figsize=(40,20))

sns.lineplot(data = cdr_logs_japan.loc[:,["Confirmed","Deaths","Recovered"]])

cdr_logs_japan.tail(1)
plt.figure(figsize=(40,20))

sns.lineplot(data = cdr_logs_skorea.loc[:,["Confirmed","Deaths","Recovered"]])
speed_per_week = pd.DataFrame([(cdr_logs_argentina.tail(7).iloc[6,4:]-cdr_logs_argentina.tail(7).iloc[0,4:])/7,

                     (cdr_logs_singapore.tail(7).iloc[6,4:]-cdr_logs_singapore.tail(7).iloc[0,4:])/7,

                     (cdr_logs_germany.tail(7).iloc[6,4:]-cdr_logs_germany.tail(7).iloc[0,4:])/7,

                     (cdr_logs_japan.tail(7).iloc[6,4:]-cdr_logs_japan.tail(7).iloc[0,4:])/7,

                     (cdr_logs_skorea.tail(7).iloc[6,4:]-cdr_logs_skorea.tail(7).iloc[0,4:])/7], 

                  index=["Argentina","Singapore","Germany","Japan","South Korea"])

sns.barplot(y=speed_per_week.iloc[:,0],x=speed_per_week.index)





sns.barplot(y=speed_per_week.iloc[:,1],x=speed_per_week.index)
sns.barplot(y=speed_per_week.iloc[:,2],x=speed_per_week.index)
cdr_logs_argentina.tail(1)

cdr_logs_germany.tail(1)

cdr_logs_japan.tail(1)

cdr_logs_singapore.tail(1)

cdr_logs_skorea.tail(1)

def lastData(df,column):

    return(df.tail(1)[column][0])

sns.barplot(y = pd.Series([lastData(cdr_logs_argentina,"Confirmed"),

lastData(cdr_logs_singapore,"Confirmed"),

lastData(cdr_logs_germany,"Confirmed"),

lastData(cdr_logs_japan,"Confirmed"),

lastData(cdr_logs_skorea,"Confirmed")]), x=["Argentina","Singapore","Germany","Japan","South Korea"])



plt.ylabel("Infectados")
population_ordered = [45195777,5850343,83783945,126476458,51269183]



sns.barplot(y = pd.Series([lastData(cdr_logs_argentina,"Confirmed"),

lastData(cdr_logs_singapore,"Confirmed"),

lastData(cdr_logs_germany,"Confirmed"),

lastData(cdr_logs_japan,"Confirmed"),

lastData(cdr_logs_skorea,"Confirmed")])*100/population_ordered, x=["Argentina","Singapore","Germany","Japan","South Korea"])

plt.ylabel("% de población infectada")
sns.barplot(y=pd.Series([lastData(cdr_logs_argentina,"Deaths"),

lastData(cdr_logs_singapore,"Deaths"),

lastData(cdr_logs_germany,"Deaths"),

lastData(cdr_logs_japan,"Deaths"),

lastData(cdr_logs_skorea,"Deaths")])/pd.Series([lastData(cdr_logs_argentina,"Confirmed"),

lastData(cdr_logs_singapore,"Confirmed"),

lastData(cdr_logs_germany,"Confirmed"),

lastData(cdr_logs_japan,"Confirmed"),

lastData(cdr_logs_skorea,"Confirmed")]), x=["Argentina","Singapore","Germany","Japan","South Korea"])

plt.ylabel("Tasa de mortalidad por infectados")
sns.barplot(y=pd.Series([lastData(cdr_logs_argentina,"Recovered"),

lastData(cdr_logs_singapore,"Recovered"),

lastData(cdr_logs_germany,"Recovered"),

lastData(cdr_logs_japan,"Recovered"),

lastData(cdr_logs_skorea,"Recovered")])/pd.Series([lastData(cdr_logs_argentina,"Confirmed"),

lastData(cdr_logs_singapore,"Confirmed"),

lastData(cdr_logs_germany,"Confirmed"),

lastData(cdr_logs_japan,"Confirmed"),

lastData(cdr_logs_skorea,"Confirmed")]), x=["Argentina","Singapore","Germany","Japan","South Korea"])

plt.ylabel("Tasa de recuperación por infectados")