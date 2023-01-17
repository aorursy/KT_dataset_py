import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/country-wise-gdp-from-1994-to-2017/Country wise GDP from 1994 to 2017.csv")
df.head()
def clean_name(name):

    return name.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "per_cent").replace(".", "")
df.rename(columns=clean_name, inplace = True)
df.head()
df.groupby("country")["year"].count().sort_values(ascending = False)
temp = df.groupby("year")["country"].count().reset_index()
sns.set()

plt.figure(figsize = (20,8))

plt.title("Count of countries per year", fontdict={'fontsize' : 20})

sns.barplot(x="year", y = "country", data = temp, palette="rainbow")

plt.show()
ind_gdp = df[df["country"] == "India"]
sns.set()

plt.figure(figsize = (20,8))

plt.title("Yearwise gdp of INDIA", fontdict={'fontsize' : 25})

sns.barplot(x="year", y = "gdp_real_in_usd", data = ind_gdp)

plt.show()
sns.set()

plt.figure(figsize = (20,8))

plt.title("GDP rate of INDIA", fontdict={'fontsize' : 25})

sns.lineplot(x="year", y = "gdp_change_per_cent", data = ind_gdp)

plt.show()
ind_gdp["gdp_in_usd"].std()