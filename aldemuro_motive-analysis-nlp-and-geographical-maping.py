import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import collections



pd.options.display.max_columns = 999



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')
terror=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')

terror.rename(columns={'country_txt':'Country','motive':'Motive'},inplace=True)

#terror=terror[['Country','Motive']]

#terror['casualities']=terror['Killed']+terror['Wounded']

terror.head(3)
plt.subplots(figsize=(15,6))

sns.countplot('Year',data=terror,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities Each Year')

plt.show()
import nltk

from wordcloud import WordCloud, STOPWORDS

motive=terror['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')

words=nltk.tokenize.word_tokenize(motive)

word_dist = nltk.FreqDist(words)

stopwords = nltk.corpus.stopwords.words('english')

words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 

wordcloud = WordCloud(stopwords=STOPWORDS).generate(" ".join(words_except_stop_dist))

plt.imshow(wordcloud)

fig=plt.gcf()

fig.set_size_inches(10,16)

plt.axis('off')

plt.show(wordcloud)
terror = terror.replace("United Kingdom","United Kingdom of Great Britain and Northern Ireland")



df = pd.DataFrame(terror.groupby('Country')['Country'].count())

df.columns = ['count']

df.index.names = ['Country']

df = df.reset_index()

df.head(10)
from iso3166 import countries

import iso3166

#countries.get(dftotal['Country'])

countlist= pd.DataFrame(iso3166.countries_by_alpha3).T.reset_index()



countlist = countlist[[0,2]]

countlist.rename(columns={0:'Country',2:'code'},inplace=True)

countlist.head(10)
dftotal = pd.merge(df, countlist, on=['Country', 'Country'])

dftotal.head(10)
data = dict(type='choropleth',

            locations=dftotal['code'],

            text=dftotal['Country'],

            z=dftotal['count'],

            ) 



layout = dict(

    title = 'the spread of terrorist activity in the world from year 1970 to year 2016',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)





choromap = go.Figure(data=[data], layout=layout)

py.iplot( choromap, filename='d3' )