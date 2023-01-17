!pip install hazm

!pip install https://github.com/sobhe/hazm/archive/master.zip --upgrade

!pip install wordcloud-fa
import pandas as pd

import numpy as np

from hazm import *

from wordcloud_fa import WordCloudFa

import plotly

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)

import plotly.offline as offline

import plotly.graph_objs as go
org_data = pd.read_excel('../input/persian-digikala-reviwes/2-p9vcb5bb.xlsx')

data_a = org_data[org_data['recommend'] == 'recommended']

data_b = org_data[org_data['recommend'] == 'not_recommended']

data_f = [data_a,data_b]

data = pd.concat(data_f,ignore_index = True)



data = data.dropna()



data = data[['comment','recommend','likes','dislikes']]

data.comment[4] # :D
# Preprocess Hazam



normalizer = Normalizer()

pre_data = data.copy()

pre_data['comment'] = pre_data['comment'].apply(lambda x:normalizer.normalize(x)) # Normaliz Text

pre_data['comment'] = pre_data['comment'].apply(lambda x:sent_tokenize(x))        # Sent tokeniz

pre_data['comment'] = pre_data['comment'].apply(lambda x:word_tokenize(str(x)))        # Tokenize word



print(data['comment'][4])

print('------------------------------------------------')

print(pre_data['comment'][4])

stop_words = '../input/sttopwords/stopsword.txt'
good_cm = data[data['likes'] >= 5 ]

bad_cm = data[data['dislikes'] >= 5]

bad_cm.head()
wodcloud = WordCloudFa(persian_normalize=True,stopwords=stop_words,ranks_only=True,width=1000,height=500)

wc = wodcloud.generate(str(bad_cm['comment'].values))

image = wc.to_image()

image
wodcloud = WordCloudFa(persian_normalize=True,stopwords=stop_words,ranks_only=True,width=1000,height=500)

wc = wodcloud.generate(str(good_cm['comment'].values))

image = wc.to_image()

image
print('Maximum Number of word in a Dish: ',data['comment'].str.len().max())

print('Minimum Number of Ingredients in a Dish: ',data['comment'].str.len().min())
trace = go.Histogram(

    x=pre_data['comment'].str.len(),

    xbins=dict(start=0,end=90,size=1),

   marker=dict(color='#7CFDD0'),

    opacity=0.75)

data = [trace]

layout = go.Layout(

    title='Distribution of sentence Length',

    xaxis=dict(title='Number of words'),

#     yaxis=dict(title='Count of recipes'),

    bargap=0.1,

    bargroupgap=0.2)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
longrecipes = pre_data[pre_data['comment'].str.len() > 30]

print("It seems that {} sentence consist of more than 30 word!".format(len(longrecipes)))