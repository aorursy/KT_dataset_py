import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
complete_co2_emission = pd.read_csv("/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv");

complete_co2_emission
complete_co2_emission.info()
complete_co2_emission = complete_co2_emission.drop(columns=["Code"])

complete_co2_emission.info()
plt.figure(figsize=(20,10))

plt.title("World CO2 Emission",fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )", hue="Entity",data = complete_co2_emission)
plt.figure(figsize=(20,10))

plt.title("United States Vs European Union (1750-2017)",fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="United States",data=complete_co2_emission[complete_co2_emission['Entity'] == "United States"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="EU-28",data=complete_co2_emission[complete_co2_emission['Entity'] == "EU-28"])
plt.figure(figsize=(20,10))

plt.title("CHINA Vs INDIA (1750-2017)",fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="India",data=complete_co2_emission[complete_co2_emission['Entity'] == "India"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="China",data=complete_co2_emission[complete_co2_emission['Entity'] == "China"])
plt.figure(figsize=(20,10))

plt.title("CHINA Vs INDIA Vs US Vs Russia (1750-2017)" ,fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="India",data=complete_co2_emission[complete_co2_emission['Entity'] == "India"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="China",data=complete_co2_emission[complete_co2_emission['Entity'] == "China"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="United States",data=complete_co2_emission[complete_co2_emission['Entity'] == "United States"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="Russia",data=complete_co2_emission[complete_co2_emission['Entity'] == "Russia"])
not_countries = ["World","International transport","Asia and Pacific (other)","Africa","Europe (other)","Americas (other)","Middle East","EU-28"]

new_co2_emissiongrouped = complete_co2_emission.groupby("Entity").sum()

onlyCountries = new_co2_emissiongrouped.drop(not_countries)
plt.figure(figsize=(25,10))

plt.xticks(rotation=90)

data1=onlyCountries.sort_values(by="Annual CO₂ emissions (tonnes )",ascending=False).head(100)

data1.head()

plt.title("TOTAL CO2 EMMISSION OVER ALL" ,fontsize=20)

sns.barplot(x =data1.index, y="Annual CO₂ emissions (tonnes )",data=data1)

plt.figure(figsize=(20,10))

plt.title("International Transport Alone" ,fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="International Transport",data=complete_co2_emission[complete_co2_emission['Entity'] == "International transport"])
plt.figure(figsize=(20,10))

plt.title("International Transport Vs Total World CO2 Emission" ,fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="International Transport",data=complete_co2_emission[complete_co2_emission['Entity'] == "International transport"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="World",data=complete_co2_emission[complete_co2_emission['Entity'] == "World"])

plt.figure(figsize=(20,10))

plt.title("6 Main Regions" ,fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="Asia and Pacific (other)",data=complete_co2_emission[complete_co2_emission['Entity'] == "Asia and Pacific (other)"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="Africa",data=complete_co2_emission[complete_co2_emission['Entity'] == "Africa"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="Europe (other)",data=complete_co2_emission[complete_co2_emission['Entity'] == "Europe (other)"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="Americas (other)",data=complete_co2_emission[complete_co2_emission['Entity'] == "Americas (other)"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="Middle East",data=complete_co2_emission[complete_co2_emission['Entity'] == "Middle East"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="European Union-28",data=complete_co2_emission[complete_co2_emission['Entity'] == "EU-28"])
plt.figure(figsize=(20,10))

plt.title("North Korea Vs South Korea",fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="North Korea",data=complete_co2_emission[complete_co2_emission['Entity'] == "North Korea"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="South Korea",data=complete_co2_emission[complete_co2_emission['Entity'] == "South Korea"])
plt.figure(figsize=(20,10))

plt.title("India VS Pakistan and 2 Million Deaths" ,fontsize=20)

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="India",data=complete_co2_emission[complete_co2_emission['Entity'] == "India"])

sns.lineplot(x ="Year",y="Annual CO₂ emissions (tonnes )",label="Pakistan",data=complete_co2_emission[complete_co2_emission['Entity'] == "Pakistan"])