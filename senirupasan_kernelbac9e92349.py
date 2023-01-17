# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
corona = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
corona.head()
corona.shape
corona.isna().sum()
del corona["place"]
corona.head()
corona.describe()
# getting the unique time intervals to prepare daily stats report

corona["Date"].unique()
confirmed = []

deaths = []

recovered = []



deathrate = []

recover_rate = []



for date in corona["Date"].unique():  

    

    today = corona[corona["Date"] == date]

    conf_sum = today["Confirmed"].sum()

    death_sum = today["Deaths"].sum()

    recov_sum = today["Recovered"].sum()

   

    confirmed.append(conf_sum)

    deaths.append(death_sum)

    recovered.append(recov_sum)

    

    deathrate.append(death_sum / conf_sum)

    recover_rate.append(recov_sum / conf_sum)
plt.figure(figsize=(15, 8))

plt.title("Confirmed cases over time")

plt.xticks(rotation=45, size=10)

plt.xlabel("Time")

plt.ylabel("Cases")

plt.plot(corona["Date"].unique(), confirmed)
plt.figure(figsize=(15, 8))

plt.title("Death vs Recover rates")

plt.xlabel("Time")

plt.ylabel("Rate")

plt.xticks(rotation=45, size=10)

plt.plot(corona["Date"].unique(), deathrate, label="Death rate")

plt.plot(corona["Date"].unique(), recover_rate, label="Recover rate")

plt.legend()

plt.show()
plt.figure(figsize=(15, 8))

plt.title("Comparison between cases vs Death")

plt.xlabel("Time")

plt.ylabel("Rate")

plt.xticks(rotation=45, size=10)

plt.plot(corona["Date"].unique(), confirmed, "r--", label="Cases")

plt.plot(corona["Date"].unique(), deaths, label="Death rate")

plt.legend()

plt.show()
# Countries which are affected by the virus

corona["Country"].unique()
confirmed_per_country = pd.DataFrame(columns=["country", "cases"])

confirmed_per_country["country"] = corona["Country"].unique()

confirmed_per_country = confirmed_per_country.set_index("country")

confirmed_per_country["cases"] = confirmed_per_country["cases"].fillna(0)
for country in corona["Country"].unique():

    print(country, corona[corona["Country"] == country]["Confirmed"])

    confirmed_per_country.loc[country]["cases"] = corona[corona["Country"] == country]["Confirmed"].sum()
confirmed_per_country

#corona[corona["Country"] == "Sri Lanka"]