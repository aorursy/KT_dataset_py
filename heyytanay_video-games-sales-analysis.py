! pip install -q dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport

import dabl

import warnings



from collections import Counter



warnings.simplefilter("ignore")
data = pd.read_csv("../input/videogamesales/vgsales.csv")

data.head()
data[["Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].describe()
# Before we start, let's drop all the Null values

data = data.dropna()
data['Platform'].value_counts()
# Make a Bar plot of 15 most common Platforms

platform = Counter(data['Platform'].tolist()).most_common(15)

names = [x[0] for x in platform]

counts = [x[1] for x in platform]



plt.style.use("ggplot")

sns.barplot(x=names, y=counts)

plt.title("Top 15 Platforms by Count")

plt.xlabel("Platform Name")

plt.ylabel("Count")

plt.xticks(rotation=90)

plt.show()
data["Genre"].value_counts()
# Make a barplot out of this too, so that we can see it better

genre = Counter(data["Genre"].tolist()).most_common(12)

names, values = [x[0] for x in genre], [x[1] for x in genre]



plt.style.use("classic")

sns.barplot(x=names, y=values)

plt.title("Video Game Genres")

plt.xlabel("Genre Name")

plt.ylabel("Count")

plt.xticks(rotation=90)

plt.show()
data["Publisher"].value_counts()
# Make a Bar plot of 10 most common Platforms

publisher = Counter(data['Publisher'].tolist()).most_common(15)

names = [x[0] for x in publisher]

counts = [x[1] for x in publisher]



plt.style.use("fivethirtyeight")

sns.barplot(x=names, y=counts)

plt.title("Top 10 Publishers by Count")

plt.xlabel("Publisher Name")

plt.ylabel("Count")

plt.xticks(rotation=90)

plt.show()
print(f"Average Year: {data['Year'].mean():.0f}")

print(f"75th Quantile of Column Year: {np.quantile(data['Year'], 0.75):.0f}")

print(f"99th Quantile of Column Year: {np.quantile(data['Year'], 0.99):.0f}")
# Let's make a barplot of Yearly sales

year_sales = data[['Year']].stack().value_counts().tolist()

year_list = [int(x) for x in dict(data[['Year']].stack().value_counts()).keys()]



plt.style.use("ggplot")

sns.barplot(x=year_list, y=year_sales, palette='Accent_r')

plt.xticks(rotation=55, fontsize=8)

plt.xlabel("Years")

plt.ylabel("Video Game Sales")

plt.title("Video Game Sales over the Years")

plt.show()
# Make a dataframe with Publishers arranged by Sales values

publisher_list = data['Publisher'].unique()

na_rev, eu_rev, jp_rev, ot_rev, gb_rev = [], [], [], [], []



for pub in publisher_list:

    na_rev.append(data[data['Publisher'] == pub]['NA_Sales'].sum())

    eu_rev.append(data[data['Publisher'] == pub]['EU_Sales'].sum())

    jp_rev.append(data[data['Publisher'] == pub]['JP_Sales'].sum())

    ot_rev.append(data[data['Publisher'] == pub]['Other_Sales'].sum())

    gb_rev.append(data[data['Publisher'] == pub]['Global_Sales'].sum())

    

publisher_rev = pd.DataFrame({

    'pub': publisher_list,

    'na': na_rev,

    'eu': eu_rev,

    'jp': jp_rev,

    'ot': ot_rev,

    'gb': gb_rev

})



publisher_rev.head()
# Make a dataframe with Genre arranged by Sales values

genre_list = data['Genre'].unique()

na_rev, eu_rev, jp_rev, ot_rev, gb_rev = [], [], [], [], []



for gen in genre_list:

    na_rev.append(data[data['Genre'] == gen]['NA_Sales'].sum())

    eu_rev.append(data[data['Genre'] == gen]['EU_Sales'].sum())

    jp_rev.append(data[data['Genre'] == gen]['JP_Sales'].sum())

    ot_rev.append(data[data['Genre'] == gen]['Other_Sales'].sum())

    gb_rev.append(data[data['Genre'] == gen]['Global_Sales'].sum())

    

genre_rev = pd.DataFrame({

    'genre': genre_list,

    'na': na_rev,

    'eu': eu_rev,

    'jp': jp_rev,

    'ot': ot_rev,

    'gb': gb_rev

})



genre_rev.head()
# Make a dataframe with Platforms arranged by Sales values

plt_list = data['Platform'].unique()

na_rev, eu_rev, jp_rev, ot_rev, gb_rev = [], [], [], [], []



for pl in plt_list:

    na_rev.append(data[data['Platform'] == pl]['NA_Sales'].sum())

    eu_rev.append(data[data['Platform'] == pl]['EU_Sales'].sum())

    jp_rev.append(data[data['Platform'] == pl]['JP_Sales'].sum())

    ot_rev.append(data[data['Platform'] == pl]['Other_Sales'].sum())

    gb_rev.append(data[data['Platform'] == pl]['Global_Sales'].sum())

    

plt_rev = pd.DataFrame({

    'platform': plt_list,

    'na': na_rev,

    'eu': eu_rev,

    'jp': jp_rev,

    'ot': ot_rev,

    'gb': gb_rev

})



plt_rev.head()
print(f"Average Year: {data['NA_Sales'].mean():.2f} Million")

print(f"75th Quantile of Column Year: {np.quantile(data['NA_Sales'], 0.75):.2f} Million")

print(f"99th Quantile of Column Year: {np.quantile(data['NA_Sales'], 0.99):.2f} Million")
# Let's make a barplot of Top-10 Publishers by NA Sales 

plt.style.use("classic")

sns.barplot(data=publisher_rev[:10], x='pub', y='na')

plt.xticks(rotation=90, fontsize=10)

plt.xlabel("Publishers")

plt.ylabel("Number of North American Sales")

plt.title("Top-10 North American Sales by Publishers")

plt.show()
# Let's make a barplot of Top-10 Genre by NA Sales 

plt.style.use("ggplot")

sns.barplot(data=genre_rev[:10], x='genre', y='na', palette='Accent_r')

plt.xticks(rotation=90, fontsize=10)

plt.xlabel("Genre")

plt.ylabel("Number of North American Sales")

plt.title("Top-10 North American Sales by Genre")

plt.show()
# Let's make a barplot of Top-10 Platforms by NA Sales 

plt.style.use("fivethirtyeight")

sns.barplot(data=plt_rev[:10], x='platform', y='na', palette='bone')

plt.xticks(fontsize=12)

plt.xlabel("Platform")

plt.ylabel("Number of North American Sales")

plt.title("Top-10 North American Sales by Platform")

plt.show()
print(f"Average Year: {data['EU_Sales'].mean():.2f} Million")

print(f"75th Quantile of Column Year: {np.quantile(data['EU_Sales'], 0.75):.2f} Million")

print(f"99th Quantile of Column Year: {np.quantile(data['EU_Sales'], 0.99):.2f} Million")
# Let's make a barplot of Top-10 Publishers by EU Sales 

plt.style.use("ggplot")

sns.barplot(data=publisher_rev[:10], x='pub', y='eu')

plt.xticks(rotation=90, fontsize=10)

plt.xlabel("Publishers")

plt.ylabel("Number of EU Sales")

plt.title("Top-10 EU Sales by Publishers")

plt.show()
# Let's make a barplot of Top-10 Genre by EU Sales 

plt.style.use("classic")

sns.barplot(data=genre_rev[:10], x='genre', y='eu', palette='plasma')

plt.xticks(rotation=90, fontsize=10)

plt.xlabel("Genre")

plt.ylabel("Number of EU Sales")

plt.title("Top-10 EU Sales by Genre")

plt.show()
# Let's make a barplot of Top-10 Platforms by EU Sales 

plt.style.use("fivethirtyeight")

sns.barplot(data=plt_rev[:10], x='platform', y='eu')

plt.xticks(fontsize=12)

plt.xlabel("Platform")

plt.ylabel("Number of EU Sales")

plt.title("Top-10 EU Sales by Platform")

plt.show()
print(f"Average Year: {data['Global_Sales'].mean():.2f} Million")

print(f"75th Quantile of Column Year: {np.quantile(data['Global_Sales'], 0.75):.2f} Million")

print(f"99th Quantile of Column Year: {np.quantile(data['Global_Sales'], 0.99):.2f} Million")
# Let's make a barplot of Top-10 Publishers by Global Sales

plt.style.use("classic")

sns.barplot(data=publisher_rev[:10], x='pub', y='gb', palette='Wistia')

plt.xticks(rotation=90, fontsize=10)

plt.xlabel("Publishers")

plt.ylabel("Number of Global Sales")

plt.title("Top-10 Global Sales by Publishers")

plt.show()
# Let's make a barplot of Top-10 Genre by Global Sales 

plt.style.use("fivethirtyeight")

sns.barplot(data=genre_rev[:10], x='genre', y='gb', palette='rainbow')

plt.xticks(rotation=90, fontsize=10)

plt.xlabel("Genre")

plt.ylabel("Number of Global Sales")

plt.title("Top-10 Global Sales by Genre")

plt.show()
# Let's make a barplot of Top-10 Platforms by EU Sales 

plt.style.use("ggplot")

sns.barplot(data=plt_rev[:10], x='platform', y='gb')

plt.xticks(fontsize=12)

plt.xlabel("Platform")

plt.ylabel("Number of Global Sales")

plt.title("Top-10 Global Sales by Platform")

plt.show()
# Make a list of Top-10 Games by revenue

top10gb = data['Global_Sales'][:10]

top10gamesgb = []

for pr in top10gb:

    top10gamesgb.append(data[data['Global_Sales'] == pr]['Name'].tolist()[0])



print(f"Top-10 Games by Global Sales are:\n{top10gamesgb}")
# Let's Plot the top 10 games in world by revenue



plt.style.use("fivethirtyeight")

sns.barplot(x=top10gb, y=top10gamesgb)

plt.title("Top 10 Games in World by Revenue")

plt.xlabel("Global Sales (in millions)")

plt.ylabel("Games")

plt.show()
# Make a list of Top-10 Games by revenue in North America

top10na = data['NA_Sales'][:10]

top10gamesna = []

for pr in top10na:

    top10gamesna.append(data[data['NA_Sales'] == pr]['Name'].tolist()[0])



print(f"Top-10 Games by North American Sales are:\n{top10gamesna}")
# Let's Plot the top 10 games in NA by revenue



plt.style.use("classic")

sns.barplot(x=top10na, y=top10gamesna)

plt.title("Top 10 Games in North America by Revenue")

plt.xlabel("NA Sales (in millions)")

plt.ylabel("Games")

plt.show()
# Make a list of Top-10 Games by revenue in European Union

top10eu = data['EU_Sales'][:10]

top10gameseu = []

for pr in top10eu:

    top10gameseu.append(data[data['EU_Sales'] == pr]['Name'].tolist()[0])



print(f"Top-10 Games by European Union Sales are:\n{top10gameseu}")
# Let's Plot the top 10 games in EU by revenue



plt.style.use("ggplot")

sns.barplot(x=top10eu, y=top10gameseu)

plt.title("Top 10 Games in European Union by Revenue")

plt.xlabel("EU Sales (in millions)")

plt.ylabel("Games")

plt.show()
# Pairplot with raw data

plt.figure(figsize=(16, 9))

sns.pairplot(data=data)

plt.show()
# Pairplot with correlation

plt.figure(figsize=(16, 9))

sns.pairplot(data=data.corr())

plt.show()
# Correlation Heatmap

sns.heatmap(data.corr(), annot=True)
dabl.plot(data, target_col='Global_Sales')
profile = ProfileReport(df=data, title="Video Game Data Profile Report")

profile.to_notebook_iframe()