import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns





# for visualization

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)
print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))
df.info()
df.rename(columns={'  num_pages': 'number_pages'}, inplace=True)
df.shape
df.head()
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce') 
df['publication_year'] = df['publication_date'].dt.year
df['publisher_count'] = df.groupby('publisher')['publisher'].transform('count')
df_major_publisher = df.loc[(df['publisher_count'] < 1300) & (df['publisher_count'] >= 80)]
corrMatrix=df[["ratings_count","text_reviews_count","publication_year","average_rating",

"number_pages","publisher_count"]].corr()

sns.set(font_scale=1.10)

plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,

square=True,annot=True,cmap='viridis',linecolor="white")

plt.title('Correlation between features');
corr_matrix = df.corr()

corr_matrix["ratings_count"].sort_values(ascending=False)
plt.title('Major publisher')

df_major_publisher['publisher'].value_counts().plot.bar(color=('r', 'g', 'c', 'b', 'y'),figsize=(15,10))
import plotly.graph_objs as go



def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace

cnt_srs = df_major_publisher.groupby('publisher')['ratings_count'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace0 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(50, 71, 96, 0.6)')

layout = go.Layout(title = 'The rating count by major publishers', width=1000, height=700)

fig = go.Figure(data = trace0, layout = layout)

fig
cnt_srs = df['average_rating'].value_counts()

trace1 = go.Scatter(

    x = cnt_srs.index,

    y = cnt_srs.values,

    mode = "markers",

    marker = dict(color = 'rgba(200, 50, 55, 0.8)')

)

data = [trace1]

layout = dict(title = 'Average rating',

xaxis= dict(title= 'The ratings',ticklen= 5,zeroline= False)

)

fig = go.Figure(data = data, layout = layout)

fig
df['major_publisher'] = np.where(df.publisher_count > 100, '1','0')
df.info()
from plotly.offline import init_notebook_mode, iplot

cnt_ = df['major_publisher'].value_counts()

fig = {

"data": [

{

"values": cnt_.values,

"labels": cnt_.index,

"domain": {"x": [0, .5]},

"name": "Train types",

"hoverinfo":"label+percent+name",

"hole": .7,

"type": "pie"

},],

"layout": {

"title":"The ratio of a major publisher",

"annotations": [

{ "font": { "size": 20},

"showarrow": False,

"text": "Pie Chart",

"x": 0.50,

"y": 1

},

]

}

}

iplot(fig)
df_minor_publisher = df

df_major_publisher = df

df_minor_publisher['minor_publisher'] = np.where(df.publisher_count < 10, 'minor','2')

df_major_publisher['major_publisher'] = np.where(df.publisher_count > 100, 'major','0')
df_minor_publisher['minor_publisher'].value_counts()
df_minor_publisher = df_minor_publisher[df_minor_publisher.minor_publisher != '2']

df_major_publisher = df_major_publisher[df_major_publisher.major_publisher != '0']
df_minor_publisher.rename(columns={'minor_publisher': 'major_minor'}, inplace=True)

df_major_publisher.rename(columns={'major_publisher': 'major_minor'}, inplace=True)
df_minor_publisher = df_minor_publisher.drop(['major_publisher'], axis=1)

df_major_publisher = df_major_publisher.drop(['minor_publisher'], axis=1)
df_minor_publisher['major_minor'].value_counts()
df_major_publisher['major_minor'].value_counts()
result = df_major_publisher.append(df_minor_publisher, sort=False)
result['major_minor'].value_counts()
def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace

cnt_srs = result.groupby('major_minor')['ratings_count'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace0 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(500, 71, 96, 0.6)')

layout = go.Layout(title = 'The rating count by major and minor publishers', width=1000, height=300, xaxis_title="The rating count",)

fig = go.Figure(data = trace0, layout = layout)

fig
import plotly.graph_objects as go



X = df['average_rating']

Y = df['ratings_count']



layout = go.Layout(

    autosize=False,

    title = 'The correlation between average rating and ratings count',

    width=1000,

    height=700)

fig = go.Figure([go.Bar(x=X, y=Y)], layout = layout)

fig.show()
df[['average_rating', 'text_reviews_count']].groupby(['text_reviews_count'], as_index=False).mean().sort_values(by='text_reviews_count', ascending=False)
data = pd.concat([df['text_reviews_count'], df['ratings_count']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.title('The rating counts by text reviews count')

plt.xlabel('text_reviews_count')

plt.ylabel('ratings_count')

fig = plt.scatter(x='text_reviews_count', y="ratings_count", data=data)
pd.concat([df['number_pages'], df['ratings_count']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.title('The rating counts by number of pages')

fig = sns.regplot(x='number_pages', y="ratings_count", data=df)
data = pd.concat([df['number_pages'], df['text_reviews_count']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.title('The text review count by number of pages')

fig = plt.scatter(x='number_pages', y="text_reviews_count", data=data)
df.loc[(df['publication_year'] <= 2000) & (df['ratings_count'] >= 900000)] 
var = df[(df['publication_year'] <= 2000) & (df['ratings_count'] >= 900000)].sort_values("publication_date", ascending=False).groupby("publication_date").std()

mean = df[(df['publication_year'] <= 2000) & (df['ratings_count'] >= 900000)].sort_values("publication_date", ascending=False).groupby("publication_date").mean()
df.info()
mean["ratings_count"].plot(yerr=var["ratings_count"],ylim=(0,3000000))
fre = df['publication_year'].mode()[0]

df['publication_year'] = df['publication_year'].fillna(fre)
df['publication_year'] = df['publication_year'].astype(int)
cut_labels_4 = ['~1920','1920~1940','1940~1960','1960~1980','1980~2000','2000~2020']

cut_bins = [0, 1920, 1940, 1960, 1980, 2000, 2020]

df['publication_year_band'] = pd.cut(df['publication_year'], bins=cut_bins, labels=cut_labels_4)
plt.title('Publication years of books')

df['publication_year_band'].value_counts().plot.bar(color=('r', 'g', 'c', 'b', 'y'))
def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace

cnt_srs = df.groupby('publication_year_band')['ratings_count'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace0 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(50, 71, 206, 0.6)')

layout = go.Layout(title = 'The rating counts by publication years', width=1000, height=700)

fig = go.Figure(data = trace0, layout = layout)

fig