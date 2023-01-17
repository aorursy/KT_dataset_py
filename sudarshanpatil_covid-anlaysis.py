import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/covid19-deaths-in-the-us/COVID-19_Death_Counts_by_Sex__Age__and_State.csv")
df.head()

df1 = df
df.shape 
df.dtypes
num_col = df[df.dtypes[df.dtypes=='float64'].reset_index()["index"]]

df[num_col.columns.to_list()] = num_col.fillna(num_col.median())
df.isna().sum()
df.head()
def clean_name(name):

    return name.lower().strip().replace(" ", "_").replace(",", "_").replace("-", "_")
df.rename(columns=clean_name, inplace=True)
df.head()
sns.set()

plt.title("Gender representation of COVID 19 patients", fontdict={'fontsize': 20})

sns.countplot(df.sex, palette="inferno")

plt.show()
sns.set()

plt.title("Age Group wise representation of COVID 19 patients", fontdict={'fontsize': 20})

sns.countplot(y=df.age_group, palette="plasma")

plt.show()
sns.set()

plt.figure(figsize =  (10,6))

plt.title("Realationship betwen COVID 19 - pneumonia patients death", fontdict={'fontsize': 15})

sns.scatterplot(x = "covid_19_deaths", y = "pneumonia_deaths", data= df, palette="inferno", hue = "sex")

plt.show()
state = df.groupby("state")["covid_19_deaths"].count().reset_index().sort_values(by = "covid_19_deaths", ascending= False)[:5]
sns.set()

plt.figure(figsize =  (10,6))

plt.title("Top 5 most infected states in US", fontdict={'fontsize': 20})

sns.barplot(x = "state", y ="covid_19_deaths", data = state, palette="coolwarm")

plt.show()