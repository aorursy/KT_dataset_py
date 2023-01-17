# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/co2-ghg-emissionsdata/co2_emission.csv")
df.sample(10)
df.rename(columns={"Entity":"Country", "Annual CO₂ emissions (tonnes )": "CO2 Emission (tonnes)"}, inplace=True)
df.info()
df["Country"] = df["Country"].astype("category") # Do it to decrease memory usage
df.shape
df.info()
def missing_value_of_data(df):
    total=df.isnull().sum().sort_values(ascending=False)
    percentage=round(total/df.shape[0]*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

missing_value_of_data(df)
df["Country"].unique()
eu = ["Austria","Belgium","Bulgaria","Croatia","Cyprus","Czech Republic",
          "Denmark","Estonia","Finland","France","Germany","Greece","Hungary",
          "Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands",
          "Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden"]
middle_east = ["United Arab Emirates", "Turkey", "Saudi Arabia", "Iran", "Iraq", "Israel", "Yemen", "Qatar"]
df_eu = df[df["Country"] == eu[0]]
for i in range(1, len(eu)):
    df_eu = df_eu.append(df[df["Country"] == eu[i]])

df_middleast = df[df["Country"] == middle_east[0]]
for i in range(1,len(middle_east)):
    df_middleast = df_middleast.append(df[df["Country"] == middle_east[i]])
eu_total = df_eu["CO2 Emission (tonnes)"][df_eu["Year"] >= 2010].sum()
usa_total = df["CO2 Emission (tonnes)"][(df["Country"] == "United States") & (df["Year"] > 2010)].sum()
china_total = df["CO2 Emission (tonnes)"][(df["Country"] == "China") & (df["Year"] > 2010)].sum()
india_total = df["CO2 Emission (tonnes)"][(df["Country"] == "India") & (df["Year"] > 2010)].sum()
russia_total = df["CO2 Emission (tonnes)"][(df["Country"] == "Russia") & (df["Year"] > 2010)].sum()
japan_total = df["CO2 Emission (tonnes)"][(df["Country"] == "Japan") & (df["Year"] > 2010)].sum()
middleast_total = df_middleast["CO2 Emission (tonnes)"][df_middleast["Year"] >= 2010].sum()
countries_total = {"EU":[eu_total], "USA":[usa_total],
                   "China":[china_total], "India":[india_total], "Russia":[russia_total],
                   "Japan":[japan_total], "Middle East":[middleast_total]}
columns = ["EU", "USA", "China", "India", "Russia", "Japan", "Middle East"]
df2 = pd.DataFrame(data=countries_total)
df2 = df2.transpose()
df2
df2.rename(columns={0:"Total CO2 Emission"}, inplace=True)
df2["Total CO2 Emission"].sort_values(ascending=False).plot(kind="bar")
def barchart(df):
    fig = plt.figure(figsize=(10,6))
    y = df2["Total CO2 Emission"].sort_values(ascending=False)
    x_labels =[y.index[i] for i in range(len(y))]
    x =[1, 2, 3, 4, 5, 6, 7]
    plt.bar(x,y)
    plt.xticks(x,x_labels,fontsize=12)
    plt.xlabel("Countries",fontsize=12)
    plt.ylabel("Emission",fontsize=12)
    plt.title("Emission Amounts vs Places", fontsize=20)
    plt.show()

barchart(df) 
fig = px.bar(df2, x=["China", "USA", "EU", "Middle East", "India", "Russia","Japan",], y=df2["Total CO2 Emission"].sort_values(ascending=False), title="Emission Amount vs Regions")
fig.show()
df.sort_values(by="CO2 Emission (tonnes)")
df_top = df[(df["CO2 Emission (tonnes)"] > 600000000) & (df["Year"]>1990)]
drops = df_top[df_top["Country"] == "World"].index.append(df_top[df_top["Country"] == "International transport"].index)
df_top = df_top.drop(drops)
fig = px.line(df_top, x='Year', y='CO2 Emission (tonnes)', color='Country', title="Emission Amounts After 1990 by Country")
fig.show()