import pandas as pd

import matplotlib.pyplot as plt
data_path = "../input/wine-reviews/winemag-data_first150k.csv"

wine_data = pd.read_csv(data_path, index_col=0)

wine_data.head()
wine_data.describe()
wine_data.shape
wine_data.info()
plt.figure(figsize=(16,6))

wine_data['country'].value_counts().head(10).plot.bar()
plt.figure(figsize=(16,6))

(wine_data["country"].value_counts().head(10)/len(wine_data)*100).plot.bar()
plt.figure(figsize=(16,6))

(wine_data["province"].value_counts().head(20)/len(wine_data)*100).plot.bar()
plt.figure(figsize=(16,6))

wine_data['points'].value_counts().sort_index().plot.bar()

#Adjusting title, fontsize using Matplotlib

plt.title("Count of wine points", fontsize=16)

plt.xlabel("Points")

plt.ylabel("Count")
plt.figure(figsize=(16,6))

wine_data['points'].value_counts().sort_index().plot.line()

plt.xlabel("Points")

plt.ylabel("Reviews")
plt.figure(figsize=(16,6))

wine_data['points'].value_counts().sort_index().plot.area()

plt.xlabel("Points")

plt.ylabel("Reviews")
plt.figure(figsize=(16,6))

wine_data['price'].value_counts().sort_index().plot.line()

plt.xlabel("Price")

plt.ylabel("Production")
plt.figure(figsize=(16,6))

wine_data['price'].value_counts().head(100).sort_index().plot.line()

plt.xlabel("Price")

plt.ylabel("Production")
#Creating a frame of rows and columns to place the plots

fig, axi = plt.subplots(1,3, figsize = (16,3))

#plot1

wine_data['points'].value_counts().plot.bar(ax = axi[0])

axi[0].set_title("Count points of Wine")

#Plot2

wine_data['country'].value_counts().head(10).plot.bar(ax = axi[1])

axi[1].set_title('Country')

#plot3

wine_data['winery'].value_counts().head(10).plot.bar(ax = axi[2])

axi[2].set_title("No of Wines from Winery")
plt.figure(figsize=(16,6))

wine_data["points"].plot.hist()
plt.figure(figsize=(16,6))

wine_data["price"].plot.hist()
plt.figure(figsize=(16,6))

wine_data[wine_data["price"] <= 100]["price"].plot.hist()
plt.figure(figsize=(16,6))

wine_data[wine_data["price"] >= 1000]["price"].plot.hist()

wine_data[wine_data["price"] >= 1000]
plt.figure(figsize=(10,10))

wine_data["country"].value_counts().head(10).plot.pie()

plt.gca().set_aspect('equal')
plt.figure(figsize=(10,10))

wine_data["province"].value_counts().head(10).plot.pie()

plt.gca().set_aspect('equal')
plt.figure(figsize=(10,10))

wine_data["variety"].value_counts().head(10).plot.pie()

plt.gca().set_aspect('equal')
wine_data.plot.scatter(x="price", y="points", figsize=(16,6))
wine_data[wine_data["price"] <= 150].plot.scatter(x="price", y="points", figsize=(16,6))
wine_data[wine_data["price"] <= 100].plot.hexbin(x="price", y="points", gridsize=20, figsize=(16,10))