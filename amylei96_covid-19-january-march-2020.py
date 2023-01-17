# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import datetime as dt # date converter





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print("Dataset: Covid-19")

covid19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

print("Rows:", len(covid19))

print("Column Names:", ', '.join(map(str, covid19.columns)))

print("Most Recent Observation:", max(covid19["ObservationDate"]))





#conversion

covid19["ObservationDate_Coverted"] = pd.to_datetime(covid19["ObservationDate"])

covid19_Mar = covid19[covid19["ObservationDate_Coverted"].dt.month < 4]



#summary

covid19_Mar_dt_summary = covid19_Mar.groupby("ObservationDate")["ObservationDate","Confirmed","Deaths","Recovered"].sum().reset_index()

covid19_Mar_dt_summary.sort_values("ObservationDate", ascending = False).style.background_gradient(cmap="PuBu")
covid19_Mar_China = covid19_Mar[covid19_Mar["Country/Region"].str.contains("China")]

covid19_Mar_China_dt = covid19_Mar_China.groupby("ObservationDate").sum().reset_index()



plt.figure(figsize = (20,10))

plt.xticks(rotation = 90, fontsize = 15)

plt.yticks(np.arange(0, 90000, 5000), fontsize = 15)

plt.title("Cases and Deaths in China as of March 31, 2020", fontsize = 20, fontweight = "bold")

plt.plot(covid19_Mar_China_dt["ObservationDate"], covid19_Mar_China_dt["Confirmed"], "bo-")

plt.plot(covid19_Mar_China_dt["ObservationDate"], covid19_Mar_China_dt["Deaths"], "ro-")

plt.legend(["Confirmed", "Deaths"], fontsize = 20)
#Italy

covid19_Mar_Italy = covid19_Mar[covid19_Mar["Country/Region"]=="Italy"]

covid19_Mar_Italy_dt = covid19_Mar_Italy.groupby("ObservationDate").sum().reset_index()

#Iran

covid19_Mar_Iran = covid19_Mar[covid19_Mar["Country/Region"]=="Iran"]

covid19_Mar_Iran_dt = covid19_Mar_Iran.groupby("ObservationDate").sum().reset_index()

#Spain

covid19_Mar_Spain = covid19_Mar[covid19_Mar["Country/Region"]=="Spain"]

covid19_Mar_Spain_dt = covid19_Mar_Spain.groupby("ObservationDate").sum().reset_index()

#Germany

covid19_Mar_Germany = covid19_Mar[covid19_Mar["Country/Region"]=="Germany"]

covid19_Mar_Germany_dt = covid19_Mar_Germany.groupby("ObservationDate").sum().reset_index()

#United States

covid19_Mar_US = covid19_Mar[covid19_Mar["Country/Region"]=="US"]

covid19_Mar_US_dt = covid19_Mar_US.groupby("ObservationDate").sum().reset_index()

#France

covid19_Mar_France = covid19_Mar[covid19_Mar["Country/Region"]=="France"]

covid19_Mar_France_dt = covid19_Mar_France.groupby("ObservationDate").sum().reset_index()



plt.figure(figsize = (20,10))

plt.xticks(rotation = 90, fontsize = 15)

plt.yticks(np.arange(0, 200000, step = 10000), fontsize = 15)

plt.title("Cases in Most Affected Countries Outside of China as of March 31, 2020", fontsize = 20, fontweight = "bold")

plt.plot(covid19_Mar_US_dt["ObservationDate"], covid19_Mar_US_dt["Confirmed"], "o-", color = "red")

plt.plot(covid19_Mar_Italy_dt["ObservationDate"], covid19_Mar_Italy_dt["Confirmed"], "o-", color = "orange")

plt.plot(covid19_Mar_Iran_dt["ObservationDate"], covid19_Mar_Iran_dt["Confirmed"], "go-")

plt.plot(covid19_Mar_Spain_dt["ObservationDate"], covid19_Mar_Spain_dt["Confirmed"], "bo-")

plt.plot(covid19_Mar_Germany_dt["ObservationDate"], covid19_Mar_Germany_dt["Confirmed"], "o-", color = "purple")

plt.plot(covid19_Mar_France_dt["ObservationDate"], covid19_Mar_France_dt["Confirmed"], "o-", color = "black")

plt.legend(["US", "Italy", "Iran", "Spain", "Germany", "France"], fontsize = 20)
plt.figure(figsize = (20,10))

plt.xticks(rotation = 90, fontsize = 15)

plt.yticks(np.arange(0, 15000, step = 500), fontsize = 15)

plt.title("Deaths in China and Italy as of March 31, 2020", fontsize = 20, fontweight = "bold")

plt.plot(covid19_Mar_China_dt["ObservationDate"], covid19_Mar_China_dt["Deaths"], "ro-")

plt.plot(covid19_Mar_Italy_dt["ObservationDate"], covid19_Mar_Italy_dt["Deaths"], "ro--")

plt.legend(["China", "Italy"], fontsize = 20)
plt.figure(figsize = (20,10))

plt.xticks(rotation = 90, fontsize = 15)

plt.yticks(np.arange(0, 200000, step = 10000), fontsize = 15)

plt.title("Cases and Deaths in the United States as of March 31, 2020", fontsize = 20, fontweight = "bold")

plt.plot(covid19_Mar_US_dt["ObservationDate"], covid19_Mar_US_dt["Confirmed"], "bo-")

plt.plot(covid19_Mar_US_dt["ObservationDate"], covid19_Mar_US_dt["Deaths"], "ro-")

plt.legend(fontsize = 20)
covid19_Mar_US_WA = covid19_Mar_US[covid19_Mar_US["Province/State"].str.contains("Washington|WA")]

covid19_Mar_US_WA_dt = covid19_Mar_US_WA.groupby("ObservationDate").sum().reset_index()



covid19_Mar_US_CA = covid19_Mar_US[covid19_Mar_US["Province/State"].str.contains("California|CA")]

covid19_Mar_US_CA_dt = covid19_Mar_US_CA.groupby("ObservationDate").sum().reset_index()



covid19_Mar_US_NY = covid19_Mar_US[covid19_Mar_US["Province/State"].str.contains("New York|NY")]

covid19_Mar_US_NY_dt = covid19_Mar_US_NY.groupby("ObservationDate").sum().reset_index()



covid19_Mar_US_NJ = covid19_Mar_US[covid19_Mar_US["Province/State"].str.contains("New Jersey|NJ")]

covid19_Mar_US_NJ_dt = covid19_Mar_US_NJ.groupby("ObservationDate").sum().reset_index()





plt.figure(figsize = (20,10))

plt.xticks(rotation = 90, fontsize = 15)

plt.yticks(np.arange(0, 100000, step = 5000), fontsize = 15)

plt.title("Cases in Most Affected States in the United States as of March 31, 2020", fontsize = 20, fontweight = "bold")

plt.plot(covid19_Mar_US_WA_dt["ObservationDate"], covid19_Mar_US_WA_dt["Confirmed"], "bo-")

plt.plot(covid19_Mar_US_CA_dt["ObservationDate"], covid19_Mar_US_CA_dt["Confirmed"], "go-")

plt.plot(covid19_Mar_US_NY_dt["ObservationDate"], covid19_Mar_US_NY_dt["Confirmed"], "ro-")

plt.plot(covid19_Mar_US_NJ_dt["ObservationDate"], covid19_Mar_US_NJ_dt["Confirmed"], "o-", color = "orange")

plt.legend(["WA", "CA", "NY", "NJ"], fontsize = 20)