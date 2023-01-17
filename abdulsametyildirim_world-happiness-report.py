import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df15 = pd.read_csv("../input/world-happiness/2015.csv")
df16 = pd.read_csv("../input/world-happiness/2016.csv")
df17 = pd.read_csv("../input/world-happiness/2017.csv")
df18 = pd.read_csv("../input/world-happiness/2018.csv")
df19 = pd.read_csv("../input/world-happiness/2019.csv")
df15["Year"] = 2015
df16["Year"] = 2016
df17["Year"] = 2017
df18["Year"] = 2018
df19["Year"] = 2019
df15.columns
df15=df15[['Year','Country','Happiness Rank', 'Happiness Score',
       'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
       'Generosity']]
df15.columns = ['Year','Country','Happiness Rank', 'Happiness Score',
       'Economy', 'Family','Health', 'Freedom', 'Trust','Generosity']
df15.head()
df16.columns
df16 = df16[['Year','Country','Happiness Rank', 'Happiness Score',
       'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
       'Generosity']]

df16.columns = ['Year','Country','Happiness Rank', 'Happiness Score',
       'Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity']
df16.head()
df17.columns
df17 = df17[[ 'Year','Country', 'Happiness.Rank', 'Happiness.Score',
       'Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom',
       'Trust..Government.Corruption.','Generosity']]
df17.columns = ['Year','Country','Happiness Rank', 'Happiness Score',
       'Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity']
df17.head()
df18.columns
df18 = df18[['Year','Country or region','Overall rank', 'Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices','Perceptions of corruption','Generosity'
       ]]

df18.columns = ['Year','Country','Happiness Rank', 'Happiness Score',
       'Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity']
df18.head()
df19.columns
df19 = df19[['Year','Country or region','Overall rank','Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices','Perceptions of corruption','Generosity']]

df19.columns =  ['Year','Country','Happiness Rank', 'Happiness Score',
       'Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity']
df19.head()
T_15 = df15[df15["Country"]=="Turkey"]
T_16 = df16[df16["Country"]=="Turkey"]
T_17 = df17[df17["Country"]=="Turkey"]
T_18 = df18[df18["Country"]=="Turkey"]
T_19 = df19[df19["Country"]=="Turkey"]
Turkey_data = pd.concat([T_15,T_16,T_17,T_18,T_19])
Turkey_data.head()
fig, ax = plt.subplots(figsize=(8,5))
ax.set(xlabel="Years", ylabel="Happiness Score", title="Change of Happiness Score by Years")
sns.pointplot(x="Year" ,y="Happiness Score", data=Turkey_data)
plt.grid()
plt.show()
fig,ax = plt.subplots(figsize=(10,6))
ax.set(xlabel="Year", ylabel="Score")

sns.barplot(x=Turkey_data["Year"], y=Turkey_data["Economy"], color="blue", label="Economy")
sns.barplot(x=Turkey_data["Year"], y=Turkey_data["Family"], color="green", label="Family")
sns.barplot(x=Turkey_data["Year"], y=Turkey_data["Health"], color="red", label="Health")
sns.barplot(x=Turkey_data["Year"], y=Turkey_data["Freedom"], color="yellow", label="Freedom")
sns.barplot(x=Turkey_data["Year"], y=Turkey_data["Trust"], color="purple", label="Trust")
sns.barplot(x=Turkey_data["Year"], y=Turkey_data["Generosity"], color="cyan", label="generosity")

ax.legend()
plt.show()
plt.figure(figsize=(14,6))
sns.pointplot(x="Year", y="Economy", data=Turkey_data, color="blue")
sns.pointplot(x="Year", y="Family", data=Turkey_data, color="green")
sns.pointplot(x="Year", y="Health", data=Turkey_data, color="red")
sns.pointplot(x="Year", y="Freedom", data=Turkey_data, color="yellow")
sns.pointplot(x="Year", y="Trust", data=Turkey_data, color="purple")
sns.pointplot(x="Year", y="Generosity", data=Turkey_data, color="cyan")

plt.text(x=4.6, y=0.7, s="Economy",color="blue")
plt.text(x=4.6, y=0.6, s="Family",color="green")
plt.text(x=4.6, y=0.5, s="Health",color="red")
plt.text(x=4.6, y=0.4, s="Freedom",color="yellow")
plt.text(x=4.6, y=0.3, s="Trust", color="yellow")
plt.text(x=4.6, y=0.3, s="Generosity",color="cyan")

plt.grid()
plt.legend()
plt.show()
plt.figure(figsize=(14,6))
correlation = Turkey_data.drop(["Year"], axis=1).corr()
sns.heatmap(correlation, annot=True)
plt.show()
df19.head()
df19.info()
df19.isnull().sum()
plt.figure(figsize=(10,6))
plt.xticks(rotation=90)
Top_20_happiest_country = df19[0:20]
sns.barplot(x=Top_20_happiest_country["Country"], y=Top_20_happiest_country["Happiness Score"])
plt.show()
plt.figure(figsize=(10,6))
correlation_2 = Top_20_happiest_country.drop(["Year"], axis=1).corr()
sns.heatmap(correlation_2, annot=True)
plt.show()