import numpy as np 



import pandas as pd 

pd.set_option('display.max_columns', None) # To display all columns

pd.set_option("display.max_rows",100)



import matplotlib.pyplot as plt 

%matplotlib inline 



import seaborn as sns 

sns.set_style('whitegrid') 



import plotly.plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff



from wordcloud import WordCloud, STOPWORDS



data = pd.read_csv("../input/winemag-data-130k-v2.csv", low_memory=False)

data.head()
data = data.iloc[:,1:]
data.shape
# remove duplicates

data.drop_duplicates(inplace=True)

data.shape
for col in data:

    print(data[col].unique());
data.describe()
sns.regplot(data.points, data.price)



# or plt.scatter(data.points, data.price)
sns.boxplot(data.points, data.price)
data[data.price>=2000]
data.groupby("variety").agg(['mean', 'median', 'min', 'max','count'])
data.groupby("variety").mean().sort_values(by = "points", ascending = False)
data.groupby("variety").median().sort_values(by = "points", ascending = False)
data.groupby("variety").mean().sort_values(by = "price", ascending = False)
data.groupby("variety").points.count().sort_values(ascending = False)
data.groupby("variety").points.count().sort_values(ascending = False).describe()
data.groupby("taster_name").points.count().sort_values(ascending = False)
data.groupby("taster_name").points.mean().sort_values(ascending = False)
data.groupby("taster_name").points.agg(["min","max","mean","count"])
sns.boxplot(data.taster_name, data.points)

plt.xticks(rotation=90)
sns.boxplot(data.country, data.points)

plt.xticks(rotation=90)
top = data[data.points >= 95]

top.head()
bottom = data[data.points <= 85]

bottom.head()
from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(top.description))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()

wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(bottom.description))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()