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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

sns.set();
COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

covid = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
print("Total number of cases: {}".format(df.groupby('ObservationDate').sum()["Confirmed"][-1]))

print("Total number of deaths: {}".format(df.groupby('ObservationDate').sum()["Deaths"][-1]))

print("Total number of recovered: {}".format(df.groupby('ObservationDate').sum()["Recovered"][-1]))

print("Death rate to 2-dp: {}%".format(round(df.groupby('ObservationDate').sum()["Deaths"][-1]*100/df.groupby('ObservationDate').sum()["Confirmed"][-1],2)))

print("Oldest case: {}".format(df["Last Update"][0]))

print("Newest case: {}".format(df["Last Update"].iloc[-1]))

print("Number of unique countries with a case: {}".format(len(df["Country/Region"].unique())))
running_total = df.groupby('ObservationDate').sum()['Confirmed']

running_total = running_total.to_frame()

running_total["Day"] = pd.Series(range(len(running_total["Confirmed"]))).values

plt.scatter(running_total["Day"], running_total["Confirmed"])

plt.title("Cases over time")

plt.xlabel("Day")

plt.ylabel("Number of cases worldwide")
sns.regplot(x="Day", y="Confirmed", data=running_total)

plt.title("Covid-19 cases by day")
logconfirmed = np.log(running_total["Confirmed"])



plt.scatter(running_total["Day"], logconfirmed)

plt.title("Log of Cases over time")

plt.xlabel("Day")

plt.ylabel("Log of cases")
running_total["Change"] = running_total["Confirmed"].diff()

running_total["Increase"] = (running_total["Change"]/running_total["Confirmed"])*100

plt.scatter(running_total["Day"], running_total["Increase"])

plt.title("Percentage change in cases by day")

plt.xlabel("Day")

plt.ylabel("Percentage change")
deaths_total = df.groupby('ObservationDate').sum()["Deaths"].to_frame()

deaths_total["Day"] = pd.Series(range(len(deaths_total["Deaths"]))).values

sns.regplot(x="Day", y="Deaths", data=deaths_total)

plt.title("Covid-19 deaths over time")
recovered_total = df.groupby('ObservationDate').sum()["Recovered"].to_frame()

recovered_total["Day"] = pd.Series(range(len(recovered_total["Recovered"]))).values

sns.regplot(x="Day", y="Recovered", data=recovered_total)

plt.title("Covid-19 recovered over time")
confirmed_by_country = df.groupby("Country/Region").last()["Confirmed"].to_frame() #Some countries are running total, others not

sorted_countries = confirmed_by_country.sort_values(by = "Confirmed", ascending = False)

n = np.nanpercentile(sorted_countries, 90)

sorted_countries[sorted_countries["Confirmed"] > n].plot(kind = "bar", figsize = (20,5), rot = 45)

plt.title("Number of cases per country: Top 10%")

plt.ylabel("Number of cases")
confirmed_by_country = df.groupby("Country/Region").last()["Deaths"].to_frame() #Some countries are running total, others not

sorted_countries = confirmed_by_country.sort_values(by = "Deaths", ascending = False)

n = np.nanpercentile(sorted_countries, 90)

sorted_countries[sorted_countries["Deaths"] > n].plot(kind = "bar", figsize = (20,5), rot = 45)

plt.title("Number of deaths per country: Top 10%")

plt.ylabel("Number of deaths")
confirmed_by_country = df.groupby("Country/Region").last()["Recovered"].to_frame() #Some countries are running total, others not

sorted_countries = confirmed_by_country.sort_values(by = "Recovered", ascending = False)

n = np.nanpercentile(sorted_countries, 90)

sorted_countries[sorted_countries["Recovered"] > n].plot(kind = "bar", figsize = (20,5), rot = 45)

plt.title("Number of recovered patients per country: Top 10%")

plt.ylabel("Number of recovered")
covid = covid.dropna(subset = ["longitude", "latitude", "sex"])
plt.scatter(covid["latitude"], covid["longitude"])

plt.title("Latitude and longitude of cases")
m = folium.Map(location=[0, 0], zoom_start=2)



for i in range(0,len(covid)):

    folium.Circle(

      location=[covid.iloc[i]['latitude'], covid.iloc[i]['longitude']],

      radius=10000,

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(m)





folium.LayerControl().add_to(m)

m
covid["sex"] = covid["sex"].str.lower()

sns.countplot(x = "sex", data = covid)