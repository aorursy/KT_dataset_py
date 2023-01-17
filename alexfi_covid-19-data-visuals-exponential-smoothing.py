import numpy as np 

import pandas as pd

import os

from matplotlib import pyplot as plt
raw_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv", 

                     parse_dates=["ObservationDate"],

                     usecols=["ObservationDate", "Province/State", "Country/Region", "Confirmed", "Deaths", "Recovered"]

)



df = raw_df.groupby(["ObservationDate", "Country/Region"], as_index=False).sum()

date = df["ObservationDate"]

df = df.drop(columns=["ObservationDate"])

df.index = date

df.head()
country_df = df[df["Country/Region"] == "Mainland China"]



coutry_df = country_df[country_df["Confirmed"] > 0]

country_df = country_df[["Confirmed", "Deaths", "Recovered"]]

country_df.plot(figsize=(10,5))
country_df = df[df["Country/Region"] == "Germany"]



coutry_df = country_df[country_df["Confirmed"] > 0]

country_df = country_df[["Confirmed", "Deaths", "Recovered"]]

country_df.plot(figsize=(10,5))
country_df = df[df["Country/Region"] == "US"]



coutry_df = country_df[country_df["Confirmed"] > 0]

country_df = country_df[["Confirmed", "Deaths", "Recovered"]]

country_df.plot(figsize=(10,5))
country_df = df[df["Country/Region"] == "Italy"]



coutry_df = country_df[country_df["Confirmed"] > 0]

country_df = country_df[["Confirmed", "Deaths", "Recovered"]]

country_df.plot(figsize=(10,5))
country_df = df[df["Country/Region"] == "Spain"]



coutry_df = country_df[country_df["Confirmed"] > 0]

country_df = country_df[["Confirmed", "Deaths", "Recovered"]]

country_df.plot(figsize=(10,5))
df_germany = df[df["Country/Region"] == "Germany"]

df_germany = df_germany.loc["2020-03-01":]



df_germany["Confirmed"].plot(figsize=(10,5), title="Germany Confirmed Cases")
from statsmodels.tsa.api import Holt



ets_germany = Holt(df_germany["Confirmed"], exponential=True).fit()

forecast_germany = ets_germany.forecast(7)

print(forecast_germany)



df_germany["Confirmed"].plot(figsize=(10,5), color="blue")

forecast_germany.plot(kind="line", color="cyan", legend=True, label="Forecast Germany")
df_china = df[df["Country/Region"] == "Mainland China"]

df_china = df_china.drop(columns=["Country/Region"])

df_china["ActiveCases"] = df_china["Confirmed"] - df_china["Deaths"] - df_china["Recovered"]



df_china["ActiveCases"].plot(figsize=(10,5), title="China Active Cases", c="red")
df_germany["ActiveCases"] = df_germany["Confirmed"] - df_germany["Deaths"] - df_germany["Recovered"]

df_germany["ActiveCases"].plot(figsize=(10,5), title="Germany Active Cases")