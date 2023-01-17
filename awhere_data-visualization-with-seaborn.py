# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
_2015 = pd.read_csv("../input/2015.csv")
_2016 = pd.read_csv("../input/2016.csv")
_2017 = pd.read_csv("../input/2017.csv")
_2015.head()
#_2015.Region.value_counts(dropna=False)
#_2015["Happiness Score"].value_counts(dropna=False)
region = list(_2015.Region.unique())
score = []
gdp = []
for i in region:
    x = _2015[_2015.Region == i]
    score.append(x["Happiness Score"].mean())
    gdp.append(x["Economy (GDP per Capita)"].mean())

d1 = pd.DataFrame({"Region":region,"Score":score})
d1.sort_values("Score",ascending=True,inplace=True)

d2=pd.DataFrame({"Region":region, "GDP":gdp})
d2.sort_values("GDP",inplace=True,ascending=False)
#visuzalition
plt.figure(figsize=(10,7))
sns.set(style="white")
sns.barplot(x="Score", y="Region", data=d1, palette="colorblind")
#sns.despine(left = True, right = True)
plt.xlabel("Happiness Score", fontsize=18, color="blue")
plt.ylabel("Region", fontsize=18, color="blue")
plt.title("Happiness Score by Region in 2015", fontsize=18, color="red")
plt.show()
region = list(_2015.Region.unique())
economy = []
family = []
health = []
freedom = []
trust = []

for each in region:
    x = _2015[_2015.Region == each]
    economy.append(x["Economy (GDP per Capita)"].mean())
    family.append(x["Family"].mean())
    health.append(x["Health (Life Expectancy)"].mean())
    freedom.append(x["Freedom"].mean())
    trust.append(x["Trust (Government Corruption)"].mean())

#visuzalition

plt.figure(figsize = (10,5))
sns.barplot(x = economy, y = region, color = "pink", label = "Economy")
sns.barplot(x = family, y = region, color = "red", label = "Family")
sns.barplot(x = health, y = region, color = "blue", label = "Health")
sns.barplot(x = freedom, y = region, color = "orange", label = "Freedom")
sns.barplot(x = trust, y = region, color = "purple", label = "Trust")
plt.legend()
plt.show()
f,ax = plt.subplots(figsize=(15,6))
sns.heatmap(_2015.corr(), annot=True, ax=ax, lw=.7, linecolor="black", fmt=".2f")
plt.show()
sns.clustermap(_2015.corr())
plt.show()
d1.Score = d1.Score/d1.Score.max()
d2.GDP = d2.GDP/d2.GDP.max()
data = pd.concat([d1,d2.GDP],axis=1)
data.sort_values("Score",inplace=True)
data.head()
f,ax = plt.subplots(figsize=(15,5))
sns.pointplot(x="Region", y="Score",data=data, color="c")
sns.pointplot(x="Region", y="GDP",data=data, color="m")
plt.xticks(rotation=90)
plt.text(0.05,1,"Score", color="c", fontsize=18, style="italic")
plt.text(0.05,0.9,"GDP", color="m", fontsize=18, style="italic")
plt.xlabel("Region", fontsize=15, color="blue")
plt.ylabel("Values", fontsize=15, color="blue")
plt.title("Happiness Score-Economy(GDP per Capita)", fontsize=20, color="red")
plt.grid()
plt.show()
df = pd.pivot_table(_2015, index="Region", values=["Happiness Score","Freedom"])
df["Happiness Score"] = df["Happiness Score"]/max(df["Happiness Score"])  #normalize
df["Freedom"] = df["Freedom"]/max(df["Freedom"]) #normalize
sns.jointplot(x = "Happiness Score", y="Freedom", data=df, kind="kde", size=7, color="green", space=0, ratio=7)
plt.show()
plt.figure(figsize=(12,7))
sns.countplot(y = "Region", data = _2015, palette = "rainbow",)
plt.ylabel("")
plt.xlabel("")
plt.title("Number of region in data", size=20, color = "purple")
plt.show()
plt.figure(figsize=(15,7))
sns.boxplot(y="Freedom", x="Region", data=_2015, width=0.5, fliersize=7, whis=0.5, linewidth=3)
plt.ylabel("Freedom", size=18, color="blue")
plt.xlabel("")
plt.title("Freedom by Region", size=18, color="red")
plt.xticks(rotation=45)
plt.tight_layout
plt.show()
plt.figure(figsize=(20,8))
sns.violinplot(x="Region", y="Trust (Government Corruption)", data=_2015, palette="Set1", inner="points", scale="count")
plt.xticks(rotation=90)
plt.title("Percentage of Trust by Region", size=18, color="red")
plt.xlabel("Region", size=18, color="blue")
plt.ylabel("Trust (Government Corruption)", size=18,  color="blue")
plt.show()
df1 = pd.pivot_table(_2015, index="Region", values =["Happiness Score","Family"])
df1["Happiness Score"] = df1["Happiness Score"]/max(df1["Happiness Score"])
df1["Family"] = df1["Family"]/max(df1["Family"])

sns.kdeplot(df1["Happiness Score"], df1["Family"] , shade=True, cut=2, cmap="Reds", cbar=True)
plt.xlabel("Happiness Score")
plt.ylabel("Family")
plt.title("Happiness Score-Family(Social support)", size=18, color="blue")
plt.show()
sns.lmplot(y = "Happiness Score", x="Economy (GDP per Capita)", data=_2015, size=5)
plt.show()
plt.figure(figsize=(10,5))
sns.stripplot(x="Region", y="Happiness Rank", data=_2015, size=6, jitter=True)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10,5))
sns.swarmplot(x="Region", y="Happiness Rank", data=_2015, size=6)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(12,5))
sns.violinplot(x="Region", y="Happiness Rank", data=_2015,color="gray",width=5)
sns.swarmplot(x="Region", y="Happiness Rank", data=_2015, size=5, color="red")
plt.xticks(rotation=90)
plt.show()
df = _2015.iloc[:,[1,3,5,7,8]]
df.head()
sns.pairplot(df, hue="Region",size=3.5)
plt.show()
m = sns.PairGrid(_2015.iloc[:,[2,5,9]],size=3)
m.map_diag(sns.distplot)
m.map_lower(plt.scatter)
m.map_upper(sns.kdeplot)
plt.show()
sns.rugplot(df["Happiness Score"],height=0.5)
plt.grid(True)
plt.xlabel("Happiness Score")
sns.kdeplot(df["Happiness Score"], color="orange")
plt.show()
labels = _2015.Region.value_counts().index
colors = ["blue","yellow","orange","red","lavender","gray","green","brown","purple","pink"]
explode = [0,0,0,0,0,0,0,0,0,0]
sizes = _2015.Region.value_counts().values

plt.figure(figsize=(8,8))
plt.pie(sizes,explode=explode, labels=labels, colors=colors, autopct="%.2f%%")
plt.title("Number of Region")
plt.show()
turkey15 = _2015[_2015.Country == "Turkey"].iloc[:,[2,3,5,6,7,8,9]]
turkey16 = _2016[_2016.Country == "Turkey"].iloc[:,[2,3,6,7,8,9,10]]
turkey17 = _2017[_2017.Country == "Turkey"].iloc[:,[1,2,5,6,7,8,10]]
turkey17.rename(columns={"Happiness.Rank":"Happiness Rank","Happiness.Score":"Happiness Score",
                         "Economy..GDP.per.Capita.":"Economy (GDP per Capita)",
                         "Trust..Government.Corruption.":"Trust (Government Corruption)",
                         "Health..Life.Expectancy.":"Health (Life Expectancy)"}, inplace=True)
turkey = turkey15.append(turkey16)
turkey3year = turkey.append(turkey17,ignore_index=True)
turkey3year["Year"] = ["2015","2016","2017"]
plt.figure(figsize=(10,8))
sns.heatmap(turkey3year.corr(), annot=True, fmt=".2f", linewidth=2, linecolor="red")
plt.title("Correlation", size=18, color="purple")
plt.show()
#sns.swarmplot(x = "Happiness Score", y="Economy (GDP per Capita)", data=turkey3year, hue="Year",size=8)
plt.figure(figsize=(12,5))
sns.lmplot(x = "Happiness Score", y="Economy (GDP per Capita)", data=turkey3year, hue="Year", size=7)
plt.title("Happiness Score-Economy", size=18, color="purple")
plt.grid(True)
plt.show()
plt.figure(figsize=(12,5))
sns.pointplot(x = "Happiness Score", y="Freedom", data=turkey3year, hue="Year")
plt.title("Happiness Score-Freedom", size=18, color="purple")
plt.grid(True)
plt.show()
sns.pairplot(turkey3year.iloc[:,[1,3,4,7]], hue="Year",size=2)
plt.show()
