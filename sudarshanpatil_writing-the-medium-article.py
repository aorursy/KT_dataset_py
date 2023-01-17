import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/medium-articles-dataset-2020-edition/Cleaned_Medium_Data.csv")
df.head()
df.dtypes
# clean the columns name

def clean(name):

    return name.lower().strip().replace(" ", "_")

df.rename(columns = clean, inplace = True)

df.head()
df.isna().sum()
sns.set()

fig, ax = plt.subplots(figsize =(10,10))

ax = sns.countplot(y = df.publication, palette="Pastel1")

ax.set_ylabel("Categories")

ax.set_xlabel("Count of Articles")

plt.title("The Distribution of Publication", fontsize = 20)

plt.show()
sns.set()

fig, ax = plt.subplots(figsize =(10,6))

ax = sns.scatterplot (x = "reading_time", y = "claps", data = df,hue = "publication")

ax.set_xlabel("Reading time for article (in minute)")

ax.set_ylabel("Claps for the Article")

plt.title("Reading time to claps ratio", fontsize = 20)

plt.show()
df.groupby("publication")["claps"].sum().reset_index().sort_values (by = "claps", ascending = False)

# Startup publication has the highest no. of claps and  it it 
temp = df.groupby("publication")["reading_time"].mean().reset_index()

sns.set()

plt.figure(figsize = (15,6))

plt.title("Average time for the topics ", size = 20)

sns.barplot(x="publication", y  = "reading_time", data = temp, palette="Spectral")

plt.show()
plt.figure(figsize = (13,10))

sns.scatterplot(y="claps", x="title_wc", data = df, hue= "publication", size = "claps")

plt.show()
plt.figure(figsize = (8,8))

sns.scatterplot(y="reading_time", x="responses", data = df, hue= "publication")

plt.title("Reponse of people", size = 20)

plt.show()
temp = df.groupby("publication")["responses"].count().reset_index()

sns.set()

plt.figure(figsize = (15,6))

plt.title("Responses of People for the topics ", size = 20)

sns.barplot(x="publication", y  = "responses", data = temp, palette="coolwarm")

plt.show()