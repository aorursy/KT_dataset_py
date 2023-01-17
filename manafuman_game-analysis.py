import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt





df = pd.read_csv("../input/videogamesales/vgsales.csv") #read the csv file
df.head(5) #first 5 rows of the dataset
df.columns # name of all columns
df.isnull().sum() # sum of null values, as seen below Year has 271 null values and Publisher has 58
games = df.groupby('Platform').size().sort_values(ascending=False).head(20)

plt.figure(figsize=(10,8))

ax=games.plot.bar(rot=1)

plt.title("Number of games per platform")

plt.ylabel("Number of games")

plt.xlabel("Platform")

ax.set_ylim(0,2500)



for i in ax.patches:

    ax.text(i.get_x(), i.get_height()+20, str(i.get_height()),fontsize=11)
#piechart with the distribution of games based on genre



df['Genre'].value_counts()[:10].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0,0,0])

plt.title('Distribution Of Top Genres')

fig=plt.gcf()

fig.set_size_inches(7,7)

plt.show()
plt.subplots(figsize=(16,6))

sns.countplot(x ='Genre',data = df)
df.groupby('Year')['Genre'].count().plot(color='y')

fig=plt.gcf()

fig.set_size_inches(12,6)