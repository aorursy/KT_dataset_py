import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

#ignore warning messages

import warnings

warnings.filterwarnings('ignore')

import plotly

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

from wordcloud import WordCloud

df= pd.read_csv("../input/trump-tweets/trumptweets.csv")
df.head()
df.drop(['id','link'],axis=1,inplace=True)

df.shape
year=[]

month=[]

date=[]

hour=[]

minute=[]

second=[]

for x in df['date']:

    year.append(int(x.split("-")[0]))

    month.append(int(x.split("-")[1]))

    date.append(int(x.split("-")[2].split(" ")[0]))

    hour.append(int(x.split("-")[2].split(" ")[1].split(":")[0]))

    minute.append(int(x.split("-")[2].split(" ")[1].split(":")[1]))

    second.append(int(x.split("-")[2].split(" ")[1].split(":")[2]))



df['year']=year

df['month']=month

df['dates']=date

df['hour']=hour

df['minute']=minute

df['second']=second

df.drop(['date'],axis=1,inplace=True)
import re

content=[]

for tweet in df["content"]:

    content.append(re.sub('\@\S+','',tweet))
# loading stop words from nltk library

import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocessing(total_text, index, column):

    if type(total_text) is not int:

        string = ""

        #Removing link

        url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'

        total_text = re.sub(url_pattern, ' ', total_text)

        # replace every special char with space

        #total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace multiple spaces with single space

        total_text = re.sub('\s+',' ', total_text)

        #total_text=total_text.replace('realdonaldtrump','').replace('donald','').replace('trump','')

        # converting all the chars into lower-case.

        total_text = total_text.lower()

        

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from the data

            if not word in stop_words:

                word=(word)

                string += word + " "

        

        df[column][index] = string



for index, row in df.iterrows():

    if type(row['content']) is str:

        preprocessing(row['content'], index, 'content')
Category=df['hour'].value_counts().sort_index()

data = [go.Pie(

        labels = Category.index,

        values = Category.values,

        hoverinfo = 'label+value',

)]

plotly.offline.iplot(data, filename='active_category')
year_country = df['year'].value_counts().reset_index(name='counts')



fig = px.bar(year_country, x='index', y='counts',

             hover_data=['index', 'counts'], color='counts',

             labels={'label':'year v/s number'}, height=400)

fig.show()
Category=(df['year'].value_counts()/365).sort_index()

Category



from plotly.subplots import make_subplots

trace1=go.Scatter(x=Category.index,y=Category.values,mode='lines+markers',name='average tweet in a day')

data=[trace1]

layout = go.Layout(title="", height=500,width=900, legend=dict(x=0.1, y=1.1))

fig = go.Figure(data,layout=layout)

fig.show()
print(df.iloc[df['retweets'].idxmax()]['content'])

print(df.iloc[df['retweets'].idxmax()]['year'])
print(df.iloc[df['favorites'].idxmax()]['content'])

print(df.iloc[df['favorites'].idxmax()]['year'])
from collections import Counter

import re

mention_df=df.dropna(subset=["mentions"])

mentions=[]

for x in mention_df["mentions"]:

    x=x.replace("@","")

    #x=re.sub(r'[\s]+',' ',)

    if not x.strip()=="":

        mentions.append(x)



Top_ten_mentions=Counter(mentions).most_common(10)



name=[]

number=[]

for x in Top_ten_mentions:

    name.append(x[0])

    number.append(x[1])



fig = go.Figure(data=[go.Bar(x=name, y=number)])



fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(title_text='Number v/s name mentions')

fig.show()
president_date=president_year=df[((df['year']>=2017) &(df['month']>1))]

print(df.iloc[president_date['favorites'].idxmax()]['content'])

print(df.iloc[president_date['favorites'].idxmax()]['year'])
president_date=president_year=df[((df['year']>=2017) &(df['month']>1))]

print(df.iloc[president_date['retweets'].idxmax()]['content'])

print(df.iloc[president_date['retweets'].idxmax()]['year'])
business_year=df[((df['year']<=2017))]

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer().fit(business_year['content'])

bag_of_words = vec.transform(business_year['content'])

sum_words = bag_of_words.sum(axis=0) 

words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

import squarify



y =dict(words_freq[:30])



fig = plt.figure(figsize=(15, 15))

squarify.plot(sizes = y.values(), label = y.keys(), color=sns.color_palette("RdGy", n_colors=20),

             linewidth=4, text_kwargs={'fontsize':14, 'fontweight' : 'bold'})

plt.title('Top 30 words', position=(0.5, 1.0+0.03), fontsize = 20, fontweight='bold')

plt.axis('off')

plt.show()
compain_year=df[(df['year'] == 2016) & (df['month'] >5) | (df['year'] == 2017) &(df['month']==1)]

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer().fit(compain_year['content'])

bag_of_words = vec.transform(compain_year['content'])

sum_words = bag_of_words.sum(axis=0) 

words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

import squarify



y =dict(words_freq[:30])



fig = plt.figure(figsize=(15, 15))

squarify.plot(sizes = y.values(), label = y.keys(), color=sns.color_palette("RdGy", n_colors=20),

             linewidth=4, text_kwargs={'fontsize':14, 'fontweight' : 'bold'})

plt.title('Top 30 words', position=(0.5, 1.0+0.03), fontsize = 20, fontweight='bold')

plt.axis('off')

plt.show()
president_year=df[((df['year']>=2017) &(df['month']>1))]



from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer().fit(president_year['content'])

bag_of_words = vec.transform(president_year['content'])

sum_words = bag_of_words.sum(axis=0) 

words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

import squarify



y =dict(words_freq[:30])



fig = plt.figure(figsize=(15, 15))

squarify.plot(sizes = y.values(), label = y.keys(), color=sns.color_palette("RdGy", n_colors=20),

             linewidth=4, text_kwargs={'fontsize':14, 'fontweight' : 'bold'})

plt.title('Top 30 words', position=(0.5, 1.0+0.03), fontsize = 20, fontweight='bold')

plt.axis('off')

plt.show()