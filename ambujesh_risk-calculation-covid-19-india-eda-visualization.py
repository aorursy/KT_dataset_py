# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/latest-data-covid19/districts_latest.csv")

df.sample(n=10, random_state=1)
label = ['Null', 'Not-Null']

color = ['orange','lightgreen']

data = [df["Tested"].isna().sum(), len(df)-df["Tested"].isna().sum()]

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(data, colors = color, explode = explode, labels = label, shadow = True, autopct = '%.2f%%')

plt.title('Value count', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
df_mod = df[df["District"].isin(["Delhi", "Mumbai", "Bengaluru Urban", "Chennai"])]

df_mod = df_mod.reset_index()

df_mod['Date']= pd.to_datetime(df_mod['Date'])

df_mod = df_mod.drop(["index", "State"], axis=1)

df_mod.head()
label = ['Null', 'Not-Null']

color = ['orange','lightgreen']

data = [df_mod["Tested"].isna().sum(), len(df_mod)-df_mod["Tested"].isna().sum()]

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(data, colors = color, explode = explode, labels = label, shadow = True, autopct = '%.2f%%')

plt.title('Null value count', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
df_mod[df_mod["Tested"].isna()].tail(2)
df_mod = df_mod.dropna()

df_mod.head()
def risk(c,r,d,o,t):

    return ((c-r+d+o)/t)*100



def survival(c,r):

    return ((r)/c)*100



def per_increase(c,t):

    return ((c)/t)*100
df_mod["Risk"] = df_mod.apply(lambda x: risk(x["Confirmed"], x["Recovered"], x["Deceased"], x["Other"], x["Tested"]), axis=1) 

df_mod["Survival"] = df_mod.apply(lambda x: survival(x["Confirmed"], x["Recovered"]),axis=1)

df_mod["Increased"] = df_mod.apply(lambda x: per_increase(x["Confirmed"], x["Tested"]),axis=1)
df_mod.head()
def graph_city(city):

    #fig, axs = plt.subplots(ncols=1)

    plt.figure(figsize=(10,5))

    df = df_mod[df_mod["District"] == city]

    sns.lineplot(x="Date", y="Risk", data=df)

    plt.title('Covid-19 Risk percentage')

    plt.show()

    

    plt.figure(figsize=(10,5))

    sns.lineplot(x="Date", y="Survival", data=df)

    plt.title('Covid-19 Survival percentage')

    plt.show()

    

    plt.figure(figsize=(10,5))

    sns.lineplot(x="Date", y="Increased", data=df)

    plt.title('Covid-19 % increase in Cases')

    plt.show()
graph_city("Delhi")
graph_city("Bengaluru Urban")
graph_city("Chennai")
graph_city("Mumbai")