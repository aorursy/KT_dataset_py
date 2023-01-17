# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/kaggle/input/countries-of-the-world/countries of the world.csv")
data.head()
data.info()
data["Literacy (%)"] = data["Literacy (%)"].str.replace("," , ".").astype("float64")
data["Net migration"] = data["Net migration"].str.replace("," , ".").astype("float64")
data["Birthrate"] = data["Birthrate"].str.replace("," , ".").astype("float64")
data["Deathrate"] = data["Deathrate"].str.replace("," , ".").astype("float64")
data["Agriculture"] = data["Agriculture"].str.replace("," , ".").astype("float64")
data["Industry"] = data["Industry"].str.replace("," , ".").astype("float64")
data["Service"] = data["Service"].str.replace("," , ".").astype("float64")
data.info()
data.shape
data.fillna(0, inplace = True)
data.shape
#Draw heatmap for demonstrate the correlation of data

plt.figure(figsize = (12, 10))
sns.heatmap(data.corr(), annot = True)
#Demonstrate the top 30 countries which have the most populations in bar plot

mostPop30Data = data.sort_values("Population", ascending = False).head(30)

plt.figure(figsize = (10, 5))
sns.barplot(x = mostPop30Data["Country"], y = mostPop30Data["Population"])
plt.xticks(rotation = 90)
plt.title("Top 30 Countries with the Highest Populations")
plt.xlabel("Countries")
plt.ylabel("Population (in billion)")
plt.show()
#Demonstrate the top 30 countries which have the largest area in horizontal bar plot

largeArea30Data = data.sort_values("Area (sq. mi.)", ascending = False).head(30)

plt.figure(figsize = (5, 10))
sns.barplot(x = largeArea30Data["Area (sq. mi.)"], y = largeArea30Data["Country"])
plt.title("Top 30 Countries with the Largest Area")
plt.xlabel("Area")
plt.ylabel("Countries")
plt.show()
#Demonstrate the birth and death ratios of top 50 countries which have the highest literacy in point plot

highLitData = data.sort_values("Literacy (%)", ascending = False).head(50)

plt.figure(figsize = (10, 5))
sns.pointplot(x = highLitData["Country"], y = highLitData["Birthrate"], color = "green", alpha = 0.8)
sns.pointplot(x = highLitData["Country"], y = highLitData["Deathrate"], color = "red", alpha = 0.6)
plt.text(1, 32, "Birthrate", color = "green", fontsize = 14)
plt.text(1, 29, "Deathrate", color = "red", fontsize = 14)
plt.xticks(rotation = 90)
plt.title("Birth and Death Ratios of 50 Countries of Highest Literacy")
plt.xlabel("Countries")
plt.ylabel("Ratios (%)")
plt.grid()
plt.show()
#Demonstrate the birth and death ratios of top 50 countries which have the lowest literacy in point plot

lowLitData = data[data["Literacy (%)"] != 0].sort_values("Literacy (%)", ascending = True).head(50)

plt.figure(figsize = (10, 5))
sns.pointplot(x = lowLitData["Country"], y = lowLitData["Birthrate"], color = "green", alpha = 0.8)
sns.pointplot(x = lowLitData["Country"], y = lowLitData["Deathrate"], color = "red", alpha = 0.6)
plt.text(1, 5, "Birthrate", color = "green", fontsize = 14)
plt.text(1, 1, "Deathrate", color = "red", fontsize = 14)
plt.xticks(rotation = 90)
plt.title("Birth and Death Ratios of 50 Countries of Lowest Literacy")
plt.xlabel("Countries")
plt.ylabel("Ratios (%)")
plt.grid()
plt.show()
#Demonstrate the total area of each region in pie plot

regData = data.groupby("Region").sum()["Area (sq. mi.)"]

plt.figure(figsize = (7, 7))
plt.pie(regData, labels = regData.index, autopct = "%1.1f%%", shadow = True)
plt.show()
#Demonstrate the correlation between the gross domestic product (gdp) and birthrate in lm plot

plt.figure(figsize = (10, 5))
sns.lmplot(x = "GDP ($ per capita)", y = "Birthrate", data = data)
plt.xlabel("GDP")
plt.ylabel("Birthrate")
plt.title("GDP vs Birthrate")
plt.show()
#Demonstrate the countries livelihood ratios (agriculture, industry, service) in violinplot

plt.figure(figsize = (10, 5))
sns.violinplot(data = data[["Agriculture", "Industry", "Service"]])
plt.title("Countries Livelihood")
plt.show()
#Demonstrate, how population distributes over regions in swarmplot

plt.figure(figsize = (10, 5))
sns.swarmplot(x = "Region", y = "Population", data = data)
plt.xticks(rotation = 90)
plt.xlabel("Regions")
plt.ylabel("Population")
plt.title("Population distribution over regions")
plt.show()
#Demonstrate literacy over regions, by distributing countries to migration status in box plot
#Divide migrations into three categories:
#Value < 0 ---------> Sending Country
#Value > 0 ---------> Receiving Country
#Value = 0 ---------> No net migration

data["Net migration"] = data["Net migration"].apply(lambda x: "Sending Country" if x < 0 else ("Receiving Country" if x > 0 else "No Migration"))

plt.figure(figsize = (10, 5))
sns.boxplot(x = "Region", y = "Literacy (%)", hue = "Net migration", data = data)
plt.xticks(rotation = 90)
plt.xlabel("Regions")
plt.ylabel("Literacy")
plt.show()
#Demonstrate literacy over regions, by distributing countries to migration status in swarm plot

plt.figure(figsize = (10, 5))
sns.swarmplot(x = "Region", y = "Literacy (%)", hue = "Net migration", data = data)
plt.xticks(rotation = 90)
plt.xlabel("Regions")
plt.ylabel("Literacy")
plt.show()
