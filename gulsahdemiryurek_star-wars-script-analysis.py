# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt

from PIL import Image

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
episodeIV = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeIV.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)

episodeV = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeV.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)

episodeVI = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeVI.txt', delim_whitespace=True, names=["index","character","dialogue"] ,header = None)
episodeIV.drop(0,inplace=True)

episodeV.drop(0,inplace=True)

episodeVI.drop(0,inplace=True)

episodeIV.drop(["index"],axis=1,inplace=True)

episodeV.drop(["index"],axis=1,inplace=True)

episodeVI.drop(["index"],axis=1,inplace=True)
script_numIV=pd.DataFrame(episodeIV.character.value_counts()).iloc[:20]

script_numV=pd.DataFrame(episodeV.character.value_counts()).iloc[:20]

script_numVI=pd.DataFrame(episodeVI.character.value_counts()).iloc[:20]

script_numIV
trace = go.Bar(y=script_numIV.character, x=script_numIV.index,  marker=dict(color="crimson",line=dict(color='black', width=2)),opacity=0.75)

trace1 = go.Bar(y=script_numV.character,x=script_numV.index,marker=dict(color="blue",line=dict(color='black', width=2)),opacity=0.75)

trace2 = go.Bar(y=script_numVI.character, x=script_numV.index,marker=dict(color="green",line=dict(color='black', width=2)),opacity=0.75)





fig = tools.make_subplots(rows=3, cols=1,horizontal_spacing=1, subplot_titles=("A New Hope","The Empire Strikes Back","Return of The Jedi"))

 

fig.append_trace(trace, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)



fig['layout'].update(showlegend=False ,height=800,title="Number of Dialogues According to Character",paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)')





iplot(fig)
episodeIV["episode"]="A New Hope"

episodeV["episode"]="The Empire Strikes Back"

episodeVI["episode"]="Return of The Jedi"

data=pd.concat([episodeIV,episodeV,episodeVI],axis=0,ignore_index=True)
import re

import nltk

from nltk.corpus import stopwords

import nltk as nlp
description_list=[]

for description in data.dialogue:

    description=re.sub("[^a-zA-Z]", " ", description)

    description=description.lower()

    description=nltk.word_tokenize(description)

    description=[word for word in description if not word in set(stopwords.words("english"))]

    lemma=nlp.WordNetLemmatizer()

    description=[lemma.lemmatize(word) for word in description]

    description=" ".join(description)

    description_list.append(description)

data["new_script"]=description_list

data
luke=data[data.character=="LUKE"]

yoda=data[data.character=="YODA"]

han=data[data.character=="HAN"]

vader=data[data.character=="VADER"]
wave_mask_yoda = np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/yoda.png"))

wave_mask_vader= np.array(Image.open("../input/vader4/vader3.png"))

wave_mask_rebel= np.array(Image.open("../input/star-wars-movie-scripts/wordcloud_masks/rebel alliance.png"))



plt.subplots(figsize=(15,15))

stopwords= set(STOPWORDS)

wordcloud = WordCloud(mask=wave_mask_vader,background_color="black",colormap="gray" ,contour_width=2, contour_color="gray",

                      width=950,

                          height=950

                         ).generate(" ".join(vader.new_script))



plt.imshow(wordcloud ,interpolation='bilinear')

plt.axis('off')

plt.savefig('graph.png')



plt.show()
plt.subplots(figsize=(15,15))

stopwords= set(STOPWORDS)

wordcloud = WordCloud(mask=wave_mask_yoda,background_color="black",contour_width=3, contour_color="olivedrab",colormap="Greens",

                      stopwords=stopwords,   

                      width=950,

                          height=950

                         ).generate(" ".join(yoda.new_script))



plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak icin kullandigim metot

max_features = 500



count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")



sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x



print("mosltly used {} words: {}".format(max_features,count_vectorizer.get_feature_names()))

plt.subplots(figsize=(15,15))

wordcloud = WordCloud(mask=wave_mask_rebel,background_color="black",contour_width=3, contour_color="tan",colormap="rainbow",  

                      width=950,

                          height=950

                         ).generate(" ".join(count_vectorizer.get_feature_names()))



plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()