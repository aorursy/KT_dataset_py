!pip install bubbly

!pip install squarify
import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.offline as py 

import plotly.graph_objs as go

from plotly import tools

py.init_notebook_mode(connected=True)



from bubbly.bubbly import bubbleplot

import squarify



import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/googleplaystore.csv")

df.head()
null_sum = pd.DataFrame(df.isnull().sum(), columns=["Sum"])

null_percent = pd.DataFrame((df.isnull().sum() / df.shape[0]) * 100, columns=["Percent"])

total = pd.concat([null_sum, null_percent], axis=1)

total.sort_values(["Sum", "Percent"], ascending=False)
df.info()
#Droping Null Values

df = df.dropna(axis=0)
#Droping Duplicates

df.drop_duplicates("App", inplace=True)
#Removing + and , from installs

df["Installs"] = df["Installs"].apply(lambda x: x.replace('+', ''))

df["Installs"] = df["Installs"].apply(lambda x: x.replace(',', ''))



#Removing M and , and Converting size of apps to MB

df["Size"] = df["Size"].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

df["Size"] = df["Size"].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

df["Size"] = df["Size"].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

df["Size"] = df["Size"].apply(lambda x: float(str(x).replace('k', '')) / 1024 if 'k' in str(x) else x)



df["Installs"] = df["Installs"].astype("int")

df["Reviews"] = df["Reviews"].astype("int")

df.dropna(axis=0, inplace=True)

df["Size"] = df["Size"].astype("float")
#Removing $ from Price

df["Price"] = df["Price"].apply(lambda x: x.replace('$', ''))

df["Price"] = df["Price"].astype("float")
plt.style.use("_classic_test")



plt.figure(figsize=(10,8))

plt.hist(df["Rating"], bins=100);
plt.figure(figsize=(10,8))

sns.countplot(df["Installs"])

plt.xticks(rotation=90);
plt.style.use("seaborn-white")

sns.pairplot(df);
def pie_plot(cnt, colors, text):

    labels = list(cnt.index)

    values = list(cnt.values)

    

    trace = go.Pie(labels=labels, 

                   values=values, 

                   hoverinfo='value+percent', 

                   title=text, 

                   textinfo='label', 

                   hole=.4, 

                   textposition='inside', 

                   marker=dict(colors = colors,

                              ),

                  )

    return trace





types = df["Type"].value_counts()

trace = pie_plot(types, ['#09efef', '#b70333'], "Results")



py.iplot([trace], filename='Type')

df.dropna(axis=0, inplace=True)

df['ratings'] = round(df['Rating'])

df['ratings'] = df['ratings'].astype('int')

df['size'] = round(df['Size'])

df['size'] = df['size'].astype('int')





fig = bubbleplot(df, x_column='Reviews', y_column='ratings', bubble_column='App', size_column='size', color_column='Installs', x_title='Reviews', 

                 y_title='Ratings', title='Reviews vs Ratings', x_logscale=False, scale_bubble=3, height=650)



py.iplot(fig, config={'scrollzoom': True})
plt.figure(figsize=(20,20))

color = plt.cm.cool(np.linspace(0, 1, 50))



categories = df["Category"].value_counts().sort_values(ascending=True)

squarify.plot(sizes=categories.values, label=categories.index, alpha=.8, color=color)

plt.title('Tree Map For Popular Categories')

plt.axis('off')

plt.show()
genres = df["Genres"].value_counts().sort_values(ascending=True)

trace = pie_plot(genres, ["cyan", "yellow", "red"], "Genres")

py.iplot([trace])
content = df["Content Rating"].value_counts()

trace = pie_plot(content, ["yellow", "cyan", "purple", "red"], "Content")

py.iplot([trace], filename="Content")
trace = [go.Histogram(x = df.Size)]

layout = {"title": "Size"}

py.iplot({"data": trace, "layout": layout})
trace = [go.Scatter(x=df["Rating"], y=df["Size"], mode="markers")]

layout = {"title": "Ratings vs Size",

          "xaxis": {"title": "Ratings"},

          "yaxis": {"title": "Size"},

          "plot_bgcolor": "rgb(0,0,0)"} 

py.iplot({'data': trace, 'layout': layout})
# Top Categories

df[["Category", "Rating"]].groupby("Category", as_index=False).mean().sort_values('Rating', ascending=False).head(10)
plt.style.use("seaborn-white")

plt.figure(figsize=(10, 8))

sns.boxplot(x="Category", y="Rating", palette="rainbow", data=df)

plt.title("Category vs Rating")

plt.xticks(rotation=90);
# Top Genres

df[["Genres", "Rating"]].groupby("Genres", as_index=False).mean().sort_values('Rating', ascending=False).head(10)
plt.figure(figsize=(20, 10))

sns.boxplot(x="Genres", y="Rating", palette="rainbow", data=df)

plt.title("Genres vs Rating")

plt.xticks(rotation=90);
costlier = df.sort_values(by="Price", ascending=False)[["App", "Price"]].head(20)

costlier
apps = df[df["Reviews"] >= 200]

apps.head()
# Top 10 Apps 

top_apps = apps.sort_values(by=["Rating", "Reviews", "Installs"], ascending=False)[["App", "Rating", "Reviews"]].head(10)

top_apps
#APPS WITH MOST INSTALLATIONS

apps.sort_values(by="Installs", ascending=False)[["App", "Installs", "Rating"]].head(15)
#APPS WITH MOST REVIEWS

apps.sort_values(by="Reviews", ascending=False)[["App", "Reviews", "Rating"]].head(15)
reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')

reviews.head()
# Droping Null Values

reviews.dropna(axis=0, inplace=True)
from wordcloud import WordCloud

from nltk.corpus import stopwords



plt.figure(figsize=(12,12))

wc = WordCloud(background_color="white", max_words=100, colormap="rainbow", stopwords=stopwords.words("english"))

plt.imshow(wc.generate_from_text(str(reviews["Translated_Review"])), interpolation="bilinear");

plt.axis("off");