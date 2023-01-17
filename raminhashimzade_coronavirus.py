import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = ""

df = pd.read_html("https://en.wikipedia.org/wiki/Timeline_of_the_2019%E2%80%9320_Wuhan_coronavirus_outbreak",encoding="UTF8", header=0)[2]
df
df.drop("Cases", axis=1, inplace=True)

df.drop("Cases.2", axis=1, inplace=True)

df.drop("Cases.3", axis=1, inplace=True)

df.drop("Deaths + Recovered (cumulative)", axis=1, inplace=True)

df.drop("D/(D+R)", axis=1, inplace=True)

df.drop("Quarantine", axis=1, inplace=True)

df.drop("Quarantine.1", axis=1, inplace=True)

df.drop("Quarantine.2", axis=1, inplace=True)

df.drop("Quarantine.3", axis=1, inplace=True)

df.drop("Source", axis=1, inplace=True)
df.drop(df.head(4).index, inplace=True)

df.drop(df.tail(1).index, inplace=True)
df.fillna(method='ffill', inplace=True)

df.fillna(0, inplace=True)
df.rename(columns={"Date(CST)": "date", "Cases.1": "cases", "Deaths(cumulative)":"deaths", "Recovered(cumulative)":"recovered"}, inplace = True)

df.reset_index(inplace=True)

df
df["cases"]=df["cases"].astype(int)

df["deaths"]=df["deaths"].astype(int)

df["recovered"]=df["recovered"].astype(int)
df.info()
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(15,5))

df.plot(x='date', y='cases', ax=ax)
fig, ax = plt.subplots(figsize=(15,5))

df.plot(x='date', y='deaths', ax=ax, color='r')
fig, ax = plt.subplots(figsize=(15,5))

df.plot(x='date', y='recovered', ax=ax, color='g')
# VAR

from statsmodels.tsa.vector_ar.var_model import VAR

from random import random

# fit model

model = VAR(df[["cases","deaths"]])

model_fit = model.fit()

# make prediction

yhat = model_fit.forecast(model_fit.y, steps=60)
import numpy as np

 

rounded = [np.round(y) for y in yhat]

dhat = pd.DataFrame(rounded)

dhat.rename(columns={0: "infected", 1: "deaths"}, inplace = True)

dhat