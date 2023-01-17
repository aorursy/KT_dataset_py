#import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True) #offline modela ilgili

import plotly.graph_objs as go

# word cloud library

from wordcloud import WordCloud

# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read the data

book_tags = pd.read_csv('../input/goodbooks-10k/book_tags.csv',encoding="Latin1")

books = pd.read_csv('../input/goodbooks-10k/books.csv', encoding="Latin1")

ratings = pd.read_csv('../input/goodbooks-10k/ratings.csv',encoding="Latin1")

tags = pd.read_csv('../input/goodbooks-10k/tags.csv', encoding="Latin1")

to_read = pd.read_csv('../input/goodbooks-10k/to_read.csv',encoding="Latin1")
books.head(2)
books.shape
books.info()
books.describe()
books['authors'].unique()
len(books['authors'].unique())
books['average_rating'].sort_values(ascending=False)
# clean the data

books['authors'].value_counts(dropna=False)
import warnings

warnings.filterwarnings("ignore")
# the avarage reviews that given authors on goodreads.

df = books.work_text_reviews_count>29700

books_newdata2=books[df]

melted_books2= pd.melt(frame=books_newdata2,id_vars='authors',value_vars='work_text_reviews_count')

reviews_list= list(melted_books2['authors'].unique())

author_reviews_ratio=[]



for i in reviews_list:

    x= melted_books2[melted_books2['authors']==i]

    author_reviews_ratio_rate=sum(x.value)/len(x)

    author_reviews_ratio.append(  author_reviews_ratio_rate)



data = pd.DataFrame({'reviews_list':reviews_list,'author_reviews_ratio':author_reviews_ratio})

new_index = (data['author_reviews_ratio'].sort_values(ascending=False)).index.values

sorted_data2 = data.reindex(new_index)



#visualization

plt.figure(figsize=(30,20))

sns.barplot(x=sorted_data2['reviews_list'],y=sorted_data2['author_reviews_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('Authors',size='30')

plt.ylabel('Avarage Reviews')

plt.title('Avarage Reviews Given Authors',color='purple',size='30')

plt.show()
#change some features' name.

data1= books.head(20)

data1.rename(columns={'ratings_1':'R1', 'ratings_2':'R2','ratings_3':'R3','ratings_4':'R4','ratings_5':'R5'}, inplace=True)
# Percentage of Ratings According to Authors

author_list= list(data1['authors'].unique())



ratings1= []

ratings2= []

for i in author_list:

    

    x = data1[data1['authors']==i]

    ratings1.append(sum(x.R1)/len(x))

    ratings2.append(sum(x.R2)/len(x))

    

f,ax = plt.subplots(figsize = (5,8))

sns.barplot(x=ratings1,y=author_list,color='green',alpha = 0.5,label='Rating1' )

sns.barplot(x=ratings2,y=author_list,color='blue',alpha = 0.5,label='Rating2' )

ax.legend(loc='lower right',frameon = True)   

ax.set(xlabel='Percentage of Ratings', ylabel='Authors',title = "Percentage of Ratings According to Authors ")

plt.show()

# prepare data frames

df2004 = books[books.original_publication_year == 2004].iloc[:200,:]

df2005 = books[books.original_publication_year == 2005].iloc[:200,:]

df2003 = books[books.original_publication_year == 2003].iloc[:200,:]



import plotly.graph_objs as go

trace1 =go.Scatter(

                    x = data1.average_rating,

                    y = data1.R1,

                    mode = "markers",

                    name = "2004",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df2004.authors)

trace2 =go.Scatter(

                    x = data1.average_rating,

                    y = data1.R2,

                    mode = "markers",

                    name = "2005",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df2005.authors)

trace3 =go.Scatter(

                    x = data1.average_rating,

                    y = data1.R3,

                    mode = "markers",

                    name = "2003",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= df2003.authors)

data = [trace1, trace2, trace3]

layout = dict(

              xaxis= dict(title= 'Average Ratings',ticklen= 9,zeroline= False),

              yaxis= dict(title= 'Ratings',ticklen= 9,zeroline= False),

              title = "Ratings and Avarage Ratings of 2004/2005/2006"

             )

fig = dict(data = data, layout = layout)

iplot(fig)
data3=books.head(6)

sta = books.average_rating.value_counts().index[:10]  

plt.figure(figsize = (8,5))

sns.barplot(x=sta,y =books.average_rating.value_counts().values[:10]) 

plt.title('Avarage Ratings of First 10 Books ',color = 'blue',fontsize=15)

plt.show()
df = books[books.original_publication_year == 2014].iloc[:10,:]



import plotly.graph_objs as go



x = df.authors

trace1 = {

  'x': x,

  'y': df.ratings_1,

  'name': 'ratings_1',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df.ratings_2,

  'name': 'ratings_2',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 Authors'},

  'barmode': 'relative',

  'title': 'Ratings1 and Ratings2 of Top 10 Authors in 2014'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
import plotly.graph_objects as go

from plotly.subplots import make_subplots



df = books[books.original_publication_year == 2009].iloc[:7,:]

pie1 = df.ratings_1

df1 = books[books.original_publication_year == 2010].iloc[:7,:]

pie2 = df1.ratings_1

labels = df.authors.value_counts().index



# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=pie1, name="Number Of Authors Rates"),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=pie1, name="Number Of Authors Rates"),

              1, 2)

# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.45, hoverinfo="label+percent+name")

fig.update_layout(

    height=800, width=800,

    title_text="Number of Authors Ratings Rates",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='2009', x=0.175, y=0.5, font_size=14, showarrow=False),

                 dict(text='2010', x=0.82, y=0.5, font_size=14, showarrow=False)])

fig.show()
from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

df1 = books.authors[books.original_publication_year == 2009]

plt.subplots(figsize=(7,7))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(df1))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

df1 = books[books.original_publication_year == 2009]

trace0 = go.Scatter(

    y=df1.ratings_1,

    name = 'Rating1',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Scatter(

    y=df1.ratings_2,

    name = 'Rating2',

     xaxis='x2',

     yaxis='y2',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace0, trace1]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.6, 0.95],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = 'Ratings1 and Ratings2 of 2009'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
books2= books.drop(columns=['work_id','id','book_id','best_book_id','best_book_id','isbn13','original_publication_year',"image_url","small_image_url"])
books2.corr()
f,ax = plt.subplots(figsize=(7,7))

sns.heatmap(books2.corr(), annot=True, linewidths=0.6,linecolor="red", fmt= '.1f',ax=ax)

plt.show() 
from scipy import stats

data_new = pd.DataFrame({'ratings1':ratings1,'ratings2':ratings2})

g = sns.jointplot(data_new.ratings1, data_new.ratings2, kind="kde", size=5)

g = g.annotate(stats.pearsonr)

plt.show()
sns.lmplot(x="ratings_3", y="work_ratings_count", data=books2)

plt.show()