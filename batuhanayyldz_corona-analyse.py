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
# Importing libraries

import numpy as np

import pandas as pd
corona_df = pd.read_csv('/kaggle/input/corona/covid_19_data.csv')
df = corona_df.copy()

df.head() # first 5 records of coronavirus dataset.
df.tail() # last 5 records of coronavirus dataset.
df.shape
df.ndim
df.size
df.isnull().sum() # number of missing value in dataset.
# missing values of Province/State

missing_states_df = df[df["Province/State"].isnull()]

missing_states_df = missing_states_df[['ObservationDate','Country/Region','Confirmed','Deaths','Recovered']]

missing_states_df.head()
# Exists states

states_df = df[df["Province/State"].notnull()]

states_df = states_df[["ObservationDate", "Province/State", "Country/Region", "Confirmed", "Deaths", "Recovered"]]

states_df.head()
grouped_states_df = states_df.groupby(["ObservationDate", "Country/Region"])[["Confirmed", "Deaths", "Recovered"]].sum().reset_index()

first_10 = grouped_states_df.head(10)



print(f"First 10 records: \n{first_10}")

last_10 = grouped_states_df.tail(10)

print(f"Last 10 records:\n {last_10}")
# China, which is center of Coronavirus



china_df = df[df["Country/Region"] == "Mainland China"]

china_df.tail(10)
# most current records of China

lastest_china_df = china_df.groupby("Province/State").max().reset_index()

lastest_china_df.head(10)
# other countries

other_countries_df = df[~(df["Country/Region"] == "Mainland China")]

other_countries_df.head(10)
# most current records of other countries

lastest_other_countries_df = other_countries_df.groupby("Country/Region")[["ObservationDate","Confirmed", "Deaths", "Recovered"]].max().reset_index()

lastest_other_countries_df.head(10)
# most confirmed

sorted_by_confirmed_df = lastest_other_countries_df.sort_values(by = "Confirmed", ascending=False)

print(f"Top 15 countries with the highest confirmed: \n{str(sorted_by_confirmed_df.head(15))}")
sorted_by_deaths_df = lastest_other_countries_df.sort_values(by = "Deaths", ascending=False)

print(f"Top 15 countries with the highest deaths: \n{str(sorted_by_deaths_df.head(15))}")
grouped_df = df.groupby("Country/Region")[["Confirmed", "Deaths", "Recovered"]].max().reset_index()

sorted_by_deaths_with_china_df = grouped_df.sort_values(by = "Deaths", ascending=False)

print(f"Top 5 countries with the highest deaths: \n{str(sorted_by_deaths_with_china_df.head())}")
# How many countries affected by Coronavirus?

affected_by_corona_df = df["Country/Region"].nunique()

print(f"Total countries are affected by corona: {str(affected_by_corona_df)}")
# Which countries are affected by Coronavirus?

affected_countries_by_corona = df["Country/Region"].unique()

print(f"Countries are: \n {str(affected_countries_by_corona)}")
# Case number all over the world



print("Confirmed: ", df["Confirmed"].sum())

print("Deaths: ", df["Deaths"].sum())

print("Recovered: ",df["Recovered"].sum())
# any country where all patients are recovered?

grouped_df[grouped_df["Confirmed"] == grouped_df["Recovered"]]
# any country where all patients are recovered?

grouped_df[grouped_df["Confirmed"] == grouped_df["Deaths"]]
# death is more than recovered?

grouped_df[grouped_df["Deaths"] > grouped_df["Recovered"]]

# deaths ratio

grouped_df["Deaths Ratio"] = grouped_df["Deaths"] / grouped_df["Confirmed"] * 100

grouped_df.sort_values(by = "Deaths Ratio", ascending = False).head(10)
# recovered ratio

grouped_df["Recovered Ratio"] = grouped_df["Recovered"] / grouped_df["Confirmed"] * 100

grouped_df.sort_values(by = "Recovered Ratio", ascending = False).head(10)