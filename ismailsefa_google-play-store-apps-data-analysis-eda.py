import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization tools

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import seaborn as sns

import pandas_profiling as pp

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
df.sample(5)
df.info()
df.isnull().sum()
report = pp.ProfileReport(df)



report.to_file("report.html")



report
import missingno as msno

msno.matrix(df)

plt.show()
df.columns=[each.replace(" ","_") for each in df.columns]
df.columns
df["Category"]=[each.replace("_"," ") for each in df.Category]

df["Price"]=[str(each.replace("$","")) for each in df.Price]


df.Reviews = pd.to_numeric(df.Reviews, errors='coerce')

df.Price = pd.to_numeric(df.Price, errors='coerce')

df.Rating = pd.to_numeric(df.Rating, errors='coerce')
df2 = pd.DataFrame(columns = ['Category'])

df2["Category"]=[each for each in df.Category.unique()]

df2["Count"]=[len(df[df.Category==each]) for each in df2.Category]

df2=df2.sort_values(by=['Count'],ascending=False)



plt.figure(figsize=(25,15))

sns.barplot(x=df2.Category, y=df2.Count)

plt.xticks(rotation= 90)

plt.xlabel('Categorys')

plt.ylabel('Count')

plt.show()
labels = df.Android_Ver.unique()

values=[]

for each in labels:

    values.append(len(df[df.Android_Ver==each]))



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
Category1 = df[df.Category=="GAME"].Rating

Category2 = df[df.Category=="FAMILY"].Rating

Category3 = df[df.Category=="MEDICAL"].Rating



fig = go.Figure()

# Use x instead of y argument for horizontal plot

fig.add_trace(go.Box(x=Category1, name='GAME'))

fig.add_trace(go.Box(x=Category2, name='FAMILY'))

fig.add_trace(go.Box(x=Category3, name='MEDICAL'))



fig.show()
plt.subplots(figsize=(25,15))

plt.xticks(rotation=90)

ax = sns.countplot(x="Installs", data=df, palette="Set3")
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Category))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()