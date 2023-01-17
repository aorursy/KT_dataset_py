import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
data=pd.read_csv("../input/ExtractedTweets.csv")
data.head(10)
data.Tweet[0]
#import necessary libraries

import re

import nltk

from nltk.corpus import stopwords

import nltk as nlp

stopwords = stopwords.words('english')

#add some unnecessary word to stopwords list

stopwords.append("rt")

stopwords.append("u")

stopwords.append("amp")

stopwords.append("w")

stopwords.append("th")
#we created 2 different class as democrat and republican

democrat=data[data.Party=="Democrat"]

republican=data[data.Party=="Republican"]
#Cleaning democrat party tweets 

democrat_list=[]

for d in democrat.Tweet:

    d=re.sub(r'http\S+', '', d) #remove links

    d=re.sub("[^a-zA-Z]", " ", d) #remove all characters except letters

    d=d.lower() #convert all words to lowercase

    d=nltk.word_tokenize(d) #split sentences into word

    d=[word for word in d if not word in set(stopwords)] #add to stopwords list if unnecessary words.

    lemma=nlp.WordNetLemmatizer() 

    d=[lemma.lemmatize(word) for word in d] #identify the correct form of the word in the dictionary

    d=" ".join(d)

    democrat_list.append(d) #append words to list
#same process as before

republican_list=[]

for r in republican.Tweet:

    r=re.sub(r'http\S+', '', r)

    r=re.sub("[^a-zA-Z]", " ", r)

    r=r.lower()

    r=nltk.word_tokenize(r)

    r=[word for word in r if not word in set(stopwords)]

    lemma=nlp.WordNetLemmatizer()

    r=[lemma.lemmatize(word) for word in r]

    r=" ".join(r)

    republican_list.append(r)
#first 5 tweets in the list

democrat_list[0:5]
#first 5 tweets in the list

republican_list[0:5]


democrat_tweets=str(democrat_list).split()

republican_tweets=str(republican_list).split()

democrat_tweets=[word.replace("'","") for word in democrat_tweets ]

democrat_tweets=[word.replace("[", "") for word in democrat_tweets ]

democrat_tweets=[word.replace("]","") for word in democrat_tweets ]

democrat_tweets=[word.replace(",", "") for word in democrat_tweets ]



republican_tweets=[word.replace("'","") for word in republican_tweets ]

republican_tweets=[word.replace("[", "") for word in republican_tweets ]

republican_tweets=[word.replace("]","") for word in republican_tweets ]

republican_tweets=[word.replace(",", "") for word in republican_tweets ]
print("Democrat tweets word length:",len(democrat_tweets))

print("Republican tweets word length:",len(republican_tweets))
#FreqDist records the number of times each words are used. 

from nltk.probability import FreqDist

fdist_democrat = FreqDist(democrat_tweets)

fdist_republican=FreqDist(republican_tweets)
fdist_democrat
fdist_republican
import matplotlib.pyplot as plt

plt.subplots(figsize=(10,5))

fdist_democrat.plot(30,title="Democrat Tweets")

plt.subplots(figsize=(10,5))

fdist_republican.plot(30,title="Republican Tweets")
de=pd.DataFrame(list(fdist_democrat.items()), columns = ["Word","FrequencyDemocrat"])

re=pd.DataFrame(list(fdist_republican.items()), columns = ["Word","FrequencyRepublican"])

new=pd.merge(de,re,on='Word')
new.head(10)
democratclass=[] 

for each in new.FrequencyDemocrat:

    if each<50:

        democratclass.append("Very Low")

    elif 49<each<150:

        democratclass.append("Low")

    elif 149<each<500:

        democratclass.append("Medium")

    elif 499<each<1500:

        democratclass.append("High")

    else:

        democratclass.append("Very High")

        

new["democratclass"]=democratclass
republicanclass=[] 

for each in new.FrequencyRepublican:

    if each<50:

        republicanclass.append("Very Low")

    elif 49<each<150:

        republicanclass.append("Low")

    elif 149<each<500:

        republicanclass.append("Medium")

    elif 499<each<1500:

        republicanclass.append("High")

    else:

        republicanclass.append("Very High")

        

new["republicanclass"]=republicanclass
new.head()
democratveryhigh=new[new.democratclass=="Very High"]

democrathigh=new[new.democratclass=="High"]

democratmedium=new[new.democratclass=="Medium"]

democratlow=new[new.democratclass=="Low"]

democratverylow=new[new.democratclass=="Very Low"]
vhvh=democratveryhigh[new.republicanclass=="Very High"]

vhh=democratveryhigh[new.republicanclass=="High"]

vhm=democratveryhigh[new.republicanclass=="Medium"]

vhl=democratveryhigh[new.republicanclass=="Low"]

vhvl=democratveryhigh[new.republicanclass=="Very Low"]



hvh=democrathigh[new.republicanclass=="Very High"]

hh=democrathigh[new.republicanclass=="High"]

hm=democrathigh[new.republicanclass=="Medium"]

hl=democrathigh[new.republicanclass=="Low"]

hvl=democrathigh[new.republicanclass=="Very Low"]



mvh=democratmedium[new.republicanclass=="Very High"]

mh=democratmedium[new.republicanclass=="High"]

mm=democratmedium[new.republicanclass=="Medium"]

ml=democratmedium[new.republicanclass=="Low"]

mvl=democratmedium[new.republicanclass=="Very Low"]



lvh=democratlow[new.republicanclass=="Very High"]

lh=democratlow[new.republicanclass=="High"]

lm=democratlow[new.republicanclass=="Medium"]

ll=democratlow[new.republicanclass=="Low"]

lvl=democratlow[new.republicanclass=="Very Low"]



vlvh=democratverylow[new.republicanclass=="Very High"]

vlh=democratverylow[new.republicanclass=="High"]

vlm=democratverylow[new.republicanclass=="Medium"]

vll=democratverylow[new.republicanclass=="Low"]

vlvl=democratverylow[new.republicanclass=="Very Low"]
trace5 = go.Scatter(y=vhvh.FrequencyDemocrat, x=vhvh.FrequencyRepublican,text=vhvh.Word,mode='markers+text')

trace4 = go.Scatter(y=vhh.FrequencyDemocrat, x=vhh.FrequencyRepublican,text=vhh.Word,mode='markers+text')

trace3 = go.Scatter(y=vhm.FrequencyDemocrat, x=vhm.FrequencyRepublican,text=vhm.Word,mode='markers+text')

trace2 = go.Scatter(y=vhl.FrequencyDemocrat, x=vhl.FrequencyRepublican,text=vhl.Word,mode='markers+text')

trace1 = go.Scatter(y=vhvl.FrequencyDemocrat, x=vhvl.FrequencyRepublican,text=vhvl.Word,mode='markers+text')



trace10 = go.Scatter(y=hvh.FrequencyDemocrat, x=hvh.FrequencyRepublican,text=hvh.Word,mode='markers+text')

trace9 = go.Scatter(y=hh.FrequencyDemocrat, x=hh.FrequencyRepublican,text=hh.Word,mode='markers+text')

trace8 = go.Scatter(y=hm.FrequencyDemocrat, x=hm.FrequencyRepublican,text=hm.Word,mode='markers+text')

trace7 = go.Scatter(y=hl.FrequencyDemocrat, x=hl.FrequencyRepublican,text=hl.Word,mode='markers+text')

trace6 = go.Scatter(y=hvl.FrequencyDemocrat, x=hvl.FrequencyRepublican,text=hvl.Word,mode='markers+text')



trace15 = go.Scatter(y=mvh.FrequencyDemocrat, x=mvh.FrequencyRepublican,text=mvh.Word,mode='markers+text')

trace14 = go.Scatter(y=mh.FrequencyDemocrat, x=mh.FrequencyRepublican,text=mh.Word,mode='markers+text')

trace13 = go.Scatter(y=mm.FrequencyDemocrat, x=mm.FrequencyRepublican,text=mm.Word,mode='markers+text')

trace12 = go.Scatter(y=ml.FrequencyDemocrat, x=ml.FrequencyRepublican,text=ml.Word,mode='markers+text')

trace11 = go.Scatter(y=mvl.FrequencyDemocrat, x=mvl.FrequencyRepublican,text=mvl.Word,mode='markers+text')





trace20 = go.Scatter(y=lvh.FrequencyDemocrat, x=lvh.FrequencyRepublican,text=lvh.Word,mode='markers+text')

trace19 = go.Scatter(y=lh.FrequencyDemocrat, x=lh.FrequencyRepublican,text=lh.Word,mode='markers+text')

trace18 = go.Scatter(y=lm.FrequencyDemocrat, x=lm.FrequencyRepublican,text=lm.Word,mode='markers+text')

trace17 = go.Scatter(y=ll.FrequencyDemocrat, x=ll.FrequencyRepublican,text=ll.Word,mode='markers+text')

trace16 = go.Scatter(y=lvl.FrequencyDemocrat, x=lvl.FrequencyRepublican,text=lvl.Word,mode='markers+text')



trace25 = go.Scatter(y=vlvh.FrequencyDemocrat, x=vlvh.FrequencyRepublican,text=vlvh.Word,mode='markers+text')

trace24 = go.Scatter(y=vlh.FrequencyDemocrat, x=vlh.FrequencyRepublican,text=vlh.Word,mode='markers+text')

trace23 = go.Scatter(y=vlm.FrequencyDemocrat, x=vlm.FrequencyRepublican,text=vlm.Word,mode='markers+text')

trace22 = go.Scatter(y=vll.FrequencyDemocrat, x=vll.FrequencyRepublican,text=vll.Word,mode='markers+text')

trace21 = go.Scatter(y=vlvl.FrequencyDemocrat, x=vlvl.FrequencyRepublican,text=vlvl.Word,mode='markers+text')



fig = tools.make_subplots(rows=5, cols=5,shared_xaxes=True, shared_yaxes=True,print_grid=False)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 1, 4)

fig.append_trace(trace5, 1, 5)

fig.append_trace(trace6, 2, 1)

fig.append_trace(trace7, 2, 2)

fig.append_trace(trace8, 2, 3)

fig.append_trace(trace9, 2, 4)

fig.append_trace(trace10, 2, 5)

fig.append_trace(trace11, 3, 1)

fig.append_trace(trace12, 3, 2)

fig.append_trace(trace13, 3, 3)

fig.append_trace(trace14, 3, 4)

fig.append_trace(trace15, 3, 5)

fig.append_trace(trace16, 4, 1)

fig.append_trace(trace17, 4, 2)

fig.append_trace(trace18, 4, 3)

fig.append_trace(trace19, 4, 4)

fig.append_trace(trace20, 4, 5)

fig.append_trace(trace21, 5, 1)

fig.append_trace(trace22, 5, 2)

fig.append_trace(trace23, 5, 3)

fig.append_trace(trace24, 5, 4)

fig.append_trace(trace25, 5, 5)





fig['layout']['xaxis5'].update(title='Very High')

fig['layout']['xaxis4'].update(title='High')

fig['layout']['xaxis3'].update(title='Medium')

fig['layout']['xaxis2'].update(title='Low')

fig['layout']['xaxis1'].update(title='Very Low')



fig['layout']['yaxis1'].update(title='Very High')

fig['layout']['yaxis2'].update(title='High')

fig['layout']['yaxis3'].update(title='Medium')

fig['layout']['yaxis4'].update(title='Low')

fig['layout']['yaxis5'].update(title='Very Low')







fig['layout'].update(height=1500, width=1500, title= "Words in Democrat and Republican Tweets",showlegend=False, titlefont=dict(size=30),

                    annotations=[dict(showarrow=False,text="Republican Tweets",x=0.5,y=-0.06,xref="paper",yref="paper",

                                      font=dict(size=30)),dict(showarrow=False, text='Democrat Tweets',

                                                               x=-0.06,y=0.5,xref="paper",yref="paper",textangle=270,

                                                               font=dict(size=30))],

                    plot_bgcolor="snow" ,paper_bgcolor='rgb(243, 243, 243)')



iplot(fig)
from nltk.text import Text  

democrat_tweet=Text(democrat_tweets)

republican_tweet=Text(republican_tweets)
plt.subplots(figsize=(10,5))

democrat_tweet.dispersion_plot(["vote","democracy","freedom","america","american","tax","trump","clinton"])
plt.subplots(figsize=(10,5))

republican_tweet.dispersion_plot(["vote","democracy","freedom","america","american","tax","trump","clinton"])
from textblob import TextBlob

democratblob=TextBlob(str(democrat_tweets))

republicanblob=TextBlob(str(republican_tweets))
democratblob.sentiment
republicanblob.sentiment