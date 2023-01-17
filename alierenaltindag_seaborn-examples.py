import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

import os
df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
df.head()
df = df[np.isfinite(df['Rating'])]

df = df[df.Category != "1.9"]



category = list(df.Category.unique())

rate = list()

for i in category:

    x = df[df.Category == i]

    average_x = sum(x.Rating)/len(x)

    rate.append(average_x)

data = pd.DataFrame({'Category': category,'Rating':rate})

new = data.Rating.sort_values(ascending = False).index.values

sorted_data = data.reindex(new)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data["Category"],y=sorted_data.Rating)

plt.xlabel("Categories")

plt.xticks(rotation=90)

plt.ylabel("Rating")

plt.title("Category vs Rating")

plt.show()
plt.figure(figsize=(15,10))

sns.barplot(x = sorted_data["Category"],y = sorted_data["Rating"], palette = sns.cubehelix_palette(len(sorted_data)))

plt.xlabel("Categories")

plt.xticks(rotation=90)

plt.ylabel("Rating")

plt.title("Category vs Rating")

plt.show()

plt.show()
df2.head()
df2 = pd.read_csv("../input/world-happiness/2016.csv")

df2.head()
first_50 = df2.iloc[:50]



plt.figure(figsize=(20,10))

sns.pointplot(x=first_50.Country,y = first_50["Happiness Score"],color = "lime")

sns.pointplot(x=first_50.Country,y = first_50["Upper Confidence Interval"],color = "red")

plt.xlabel("Prices")

plt.xticks(rotation=90)

plt.title("Prices vs Average Rating")

plt.show()

first_50.head()
sns.jointplot(x = df2.Freedom,y = df2.Generosity,kind = "kde")

plt.show()
sns.jointplot(x = df2.Freedom,y = df2.Generosity,color = "red")

plt.show()
colors = ['grey','blue','red','yellow','green','brown', "lime", "violet", "mediumspringgreen", "blueviolet"]

explode = [0,0,0,0,0,0,0,0,0,0]

labels = list(df2.Region.unique())

size = list()

for i in labels:

    x = df2[df2.Region == i]

    new_x = sum(x["Happiness Score"])/len(x)

    size.append(new_x)

data = pd.DataFrame({"Region":labels,"Happiness Score":size})



plt.figure(figsize = (7,7))

plt.pie(data["Happiness Score"], explode = explode, labels = labels, colors = colors, autopct='%1.1f%%')

plt.title("Happiness Score according to Countries",color = "red",size = 20)

plt.show()
df2.head()
sns.lmplot(x = "Freedom",y = "Dystopia Residual", data=df2)

plt.show()
sns.kdeplot(df2["Freedom"], df2["Dystopia Residual"], shade = True, cut=3)

plt.show()
df2.head()
first_50 = df2.iloc[:,[6,8]]

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=first_50, palette=pal, inner="points")

plt.show()
data = df2.iloc[:,[7,11]]

f,ax = plt.subplots(figsize = (5, 5))

sns.heatmap(data.corr(), annot = True, linewidths = 0.5,linecolor = "black", fmt= '.1f',ax=ax)

plt.show()
query = df2[(df2.Region == "Sub-Saharan Africa") | (df2.Region == "Central and Eastern Europe")]

regions = query.Region

family = query.Family

upper_confidence = query["Upper Confidence Interval"]



new_data = pd.DataFrame({"Region":regions,"Family":family,"Upper Confidence Interval":upper_confidence})

new_data.head()
g = sns.swarmplot(x="Region",y="Family",hue="Region",data = new_data)

g.get_legend().set_visible(False)

plt.show()
data.head()
sns.pairplot(data)

plt.show()
new_data.Region.value_counts()
sns.countplot(new_data.Region)

plt.show()