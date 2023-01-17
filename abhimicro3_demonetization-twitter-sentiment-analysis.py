import numpy as np
import pandas as pd
from subprocess import check_output
from bs4 import BeautifulSoup
import re
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from nltk.corpus import stopwords
print (check_output(['ls','../input']).decode('utf8'))
tweets=pd.read_csv('../input/demonetization-tweets.csv',encoding='ISO-8859-1')
tweets.head()
tweets.shape
def clean(x):
    #Remove Html  
    x=BeautifulSoup(x).get_text()
    
    #Remove Non-Letters
    x=re.sub('[^a-zA-Z]',' ',x)
    
    #Convert to lower_case and split
    x=x.lower().split()
    
    #Remove stopwords
    stop=set(stopwords.words('english'))
    words=[w for w in x if not w in stop]
    
    #join the words back into one string
    return(' '.join(words))
tweets['text']=tweets['text'].apply(lambda x:clean(x))
tweets.head()
from nltk.sentiment import vader
from nltk.sentiment.util import *

from nltk import tokenize

sid = vader.SentimentIntensityAnalyzer()
tweets['sentiment_compound_polarity']=tweets.text.apply(lambda x:sid.polarity_scores(x)['compound'])

tweets['sentiment_negative']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neg'])
tweets['sentiment_pos']=tweets.text.apply(lambda x:sid.polarity_scores(x)['pos'])
tweets['sentiment']=''
tweets.loc[tweets.sentiment_compound_polarity>=0,'sentiment']=1

tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment']=0
tweets.head()
val=tweets['sentiment'].value_counts().reset_index()
val.columns=['Sentiment','Count']

data=[go.Bar(
  x=val.Sentiment,
y=val.Count
)]
layout=go.Layout(
    title='Demonetization Sentiment Analysis',
    xaxis=dict(title='Sentiment'),
    yaxis=dict(title='Count'))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)