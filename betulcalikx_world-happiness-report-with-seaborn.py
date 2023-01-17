import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

wh_2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")

wh_2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

wh_2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

wh_2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

wh_2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
wh_2019.head()
wh_2019.tail()
wh_2019.columns
wh_2019.describe().T
wh_2019.info()
wh_2019[wh_2019.isna().any(axis=1)]
# definition of values

country_list = list(wh_2019["Country or region"].head(25))

country_score = list(wh_2019["Score"].head(25))



# sorting

data = pd.DataFrame({"country_list": country_list, "country_score": country_score})

new_index = (data["country_score"].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# visualization

plt.figure(figsize=(15, 7))

sns.barplot(x = sorted_data["country_list"], y = sorted_data["country_score"])

plt.xticks(rotation=90)

plt.xlabel("Countries")

plt.ylabel("Scores")

plt.title("Countries and Scores")

plt.show()
# definition of values

country_list = list(wh_2019["Country or region"].head(25))

country_score = list(wh_2019["Score"].head(25))

country_GDP = list(wh_2019["GDP per capita"].head(25))

country_social_support = list(wh_2019["Social support"].head(25))

country_healthy_life = list(wh_2019["Healthy life expectancy"].head(25))

country_freedom = list(wh_2019["Freedom to make life choices"].head(25))

country_generosity = list(wh_2019["Generosity"].head(25))

country_corruption = list(wh_2019["Perceptions of corruption"].head(25))



# visualization

f, ax = plt.subplots(figsize=(12, 10))

sns.barplot(x=country_score, y=country_list, color="red", alpha=0.5, label="Score")

sns.barplot(x=country_GDP, y=country_list, color="black", alpha=1, label="GDP per capita")

sns.barplot(x=country_social_support, y=country_list, color="blue", alpha=0.7, label="Social Support")

sns.barplot(x=country_healthy_life, y=country_list, color="purple", alpha=0.9, label="Healthy life expectancy")

sns.barplot(x=country_freedom, y=country_list, color="yellow", alpha=1, label="Freedom to make life choices")

sns.barplot(x=country_generosity, y=country_list, color="pink", alpha=1, label="Generosity")

sns.barplot(x=country_corruption, y=country_list, color="cyan", alpha=0.3, label="Perceptions of corruption")



ax.legend(loc="lower right", frameon=True)

ax.set(xlabel="Survey of the state", ylabel="Countries", title="Comparison of countries according to variables")

plt.show()
# normalize 

data = pd.DataFrame({"country_list": country_list, "country_social_support": country_social_support})



new_index = (data["country_social_support"].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

sorted_data["country_social_support"] = sorted_data["country_social_support"] / max(sorted_data["country_social_support"])



data = pd.DataFrame({"country_list": country_list, "country_healthy_life": country_healthy_life})



new_index = (data["country_healthy_life"].sort_values(ascending=False)).index.values

sorted_data2 = data.reindex(new_index)

sorted_data2["country_healthy_life"] = sorted_data2["country_healthy_life"] / max(sorted_data2["country_healthy_life"])



data = pd.concat([sorted_data, sorted_data2["country_healthy_life"]], axis=1)

data.sort_values("country_social_support", inplace=True)



# visualization

f = plt.subplots(figsize=(42,10))

sns.pointplot(x="country_list", y="country_social_support", data=data,color="magenta", alpha=0.8)

sns.pointplot(x="country_list", y="country_healthy_life", data=data ,color="red", alpha=0.8)

plt.xlabel("Countries", fontsize=22, color="red")

plt.ylabel("Values", fontsize=22, color="red")

plt.text(19, 0.93, "Healthy life expectancy", color="red", fontsize=22)

plt.text(19, 0.91, "Social support", color="magenta", fontsize=22)

plt.title("Healthy life expectancy vs Social Support", fontsize=22)

plt.grid()

plt.show()
# definition of values

country_list = list(wh_2019["Country or region"])

country_score = list(wh_2019["Score"])

country_GDP = list(wh_2019["GDP per capita"])

country_social_support = list(wh_2019["Social support"])

country_healthy_life = list(wh_2019["Healthy life expectancy"])

country_freedom = list(wh_2019["Freedom to make life choices"])

country_generosity = list(wh_2019["Generosity"])

country_corruption = list(wh_2019["Perceptions of corruption"])
data = pd.DataFrame({"country_score": country_score, "country_freedom": country_freedom})



g = sns.jointplot(data.country_score, data.country_freedom, kind="kde", height=8)

plt.savefig("graph.png")

plt.show()
data = pd.DataFrame({"country_corruption": country_corruption, "country_generosity": country_generosity})



g = sns.jointplot("country_generosity", "country_corruption", data=data, height=8, ratio=4, kind="reg")
wh_2016.head()
wh_2016.Region.unique()
# nan values check

wh_2016[wh_2016.isna().any(axis=1)]
# definitions

country_region = wh_2016["Region"]

country_happiness = wh_2016["Happiness Score"]

country_lower_confidence = wh_2016["Lower Confidence Interval"]

country_upper_confidence = wh_2016["Upper Confidence Interval"]

country_economy = wh_2016["Economy (GDP per Capita)"]

country_family = wh_2016["Family"]

country_health = wh_2016["Health (Life Expectancy)"]

country_freedom = wh_2016["Freedom"]

country_trust = wh_2016["Trust (Government Corruption)"]

country_generosity = wh_2016["Generosity"]

country_dystopia = wh_2016["Dystopia Residual"]
# definitions

labels = wh_2016.Region.value_counts().index

colors = ["gray", "green", "blue", "magenta", "cyan", "yellow", "purple", "brown", "red", "orange"]

explode = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

sizes = wh_2016.Region.value_counts().values



# visualization

plt.figure(figsize=(7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%")

plt.title("Country Numbers according to Regions", color="blue", fontsize=15)

plt.show()
data = pd.DataFrame({"country_economy": country_economy, "country_happiness": country_happiness})



sns.lmplot(x="country_economy", y="country_happiness", data = data, height=8)

plt.show()
data = pd.DataFrame({"country_lower_confidence": country_lower_confidence, "country_freedom": country_freedom})



sns.kdeplot(data.country_lower_confidence, data.country_freedom, shade=True, cut=5)

plt.show()
data = pd.DataFrame({"country_economy": country_economy, "country_health": country_health})

palette = sns.light_palette((210, 90, 60), input="husl")



sns.violinplot(data=data, palette=palette, inner="points")

plt.show()
data = pd.DataFrame({"country_economy": country_economy, "country_health": country_health, "country_lower_confidence": country_lower_confidence,

                    "country_upper_confidence": country_upper_confidence})

data.corr()
f, ax = plt.subplots(figsize=(12, 7))

sns.heatmap(data.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax, cmap="BuPu")

plt.show()
data = pd.DataFrame({"country_region": country_region, "country_health": country_health})



plt.figure(figsize=(30, 7))

sns.boxplot(x="country_region", y="country_health", data=data,palette="BuPu")

plt.show()
data = pd.DataFrame({"country_region": country_region, "country_family": country_family})



plt.figure(figsize=(30, 7))

sns.swarmplot(x="country_region", y="country_family", data=data)

plt.xlabel("Region", fontsize=20, color="purple")

plt.ylabel("Family", fontsize=20, color="purple")

plt.show()
cols = wh_2016.loc[:, "Happiness Score":"Family"] # Happiness Score, Lower/Upper Confidence, Economy and Family



sns.pairplot(cols)

plt.show()
plt.figure(figsize = (30, 7))

sns.countplot(wh_2016.Region)

plt.show()
# grouping as a region and calculating sum

wh_2016.groupby("Region").sum()
# grouping as a region and calculcating economy column

economy = wh_2016.groupby("Region")[["Economy (GDP per Capita)"]].sum()

economy