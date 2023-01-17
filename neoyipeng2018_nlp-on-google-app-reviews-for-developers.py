!pip install wordcloud

!pip install transformers==2.6.0
from transformers import pipeline

import numpy as np; import pandas as pd; import random; import os

pd.set_option('display.max_colwidth', -1) #show all columns

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

%matplotlib inline
revs=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv'); revs['Translated_Review'].fillna('',inplace=True)

revs.head(2)
print(f'There are {revs.shape[0]} reviews for {len(revs.App.unique())} apps.')
revs[['App','Translated_Review','Sentiment_Polarity']].groupby('App').mean().sort_values(by="Sentiment_Polarity",ascending=False)["Sentiment_Polarity"].iloc[:20].plot.bar()
revs[['App','Sentiment_Polarity']].groupby('App').mean().sort_values(by="Sentiment_Polarity",ascending=True)["Sentiment_Polarity"].iloc[:20].plot.bar()
topIdx=revs[['App','Sentiment_Polarity']].groupby('App').mean().sort_values(by="Sentiment_Polarity",ascending=False)["Sentiment_Polarity"].iloc[:20].index

top20=revs[revs.App.isin(topIdx)].fillna('')

print(top20.shape)



text = " ".join(review for review in top20.Translated_Review)

print ("There are {} words in the combination of all reviews.".format(len(text)))
stopwords = set(STOPWORDS)

stopwords.update(["best","app","great","excellent","game","good","even","nice","awesome","love","really","sport","recipe"

                 ,"sports","recipes"]) #removing superlative words
#i took the config from https://www.datacamp.com/community/tutorials/wordcloud-python

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text) #Generate a word cloud image

plt.imshow(wordcloud, interpolation='bilinear'); plt.axis("off")
revs[revs['Translated_Review'].str.contains('easy')].sort_values(by="Sentiment_Polarity",ascending=False)['Translated_Review'].head()
revs[revs['Translated_Review'].str.contains('need')].sort_values(by="Sentiment_Polarity",ascending=False)['Translated_Review'].head()
revs[revs['Translated_Review'].str.contains('useful')].sort_values(by="Sentiment_Polarity",ascending=False)['Translated_Review'].head()
revs[revs['Translated_Review'].str.contains('time')].sort_values(by="Sentiment_Polarity",ascending=False)['Translated_Review'].head()
botIdx=revs[['App','Sentiment_Polarity']].groupby('App').mean().sort_values(by="Sentiment_Polarity",ascending=True)["Sentiment_Polarity"].iloc[:20].index

bot20=revs[revs.App.isin(botIdx)].fillna('')

print(bot20.shape)

bottext = " ".join(review for review in bot20.Translated_Review)

print ("There are {} words in the combination of all reviews.".format(len(bottext)))
stopwords = set(STOPWORDS)

stopwords.update(["best","app","great","excellent","worst","horrible","terrible","even","good","game"]) #removing superlative words
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(bottext) #Generate a word cloud image

plt.imshow(wordcloud, interpolation='bilinear'); plt.axis("off")
revs[revs['Translated_Review'].str.contains('work')].sort_values(by="Sentiment_Polarity",ascending=True)['Translated_Review'].head()
revs[revs['Translated_Review'].str.contains('time')].sort_values(by="Sentiment_Polarity",ascending=True)['Translated_Review'].head(10)
revs[revs['Translated_Review'].str.contains('load')].sort_values(by="Sentiment_Polarity",ascending=True)['Translated_Review'].head(10)
revs[revs['Translated_Review'].str.contains('play')].sort_values(by="Sentiment_Polarity",ascending=True)['Translated_Review'].head()
nlp = pipeline(task="sentiment-analysis");
Gdapp='Box'

#app=random.choice(revs.App.tolist())

print(Gdapp)



GdappRev=revs[revs.App==Gdapp]

GdappRev.Translated_Review=GdappRev.Translated_Review.astype(str)

txt=GdappRev.Translated_Review.dropna().tolist()

GdappRevTxt=' '.join(txt)
nlp(GdappRevTxt)
Ngapp='Expedia Hotels, Flights & Car Rental Travel Deals'

#app=random.choice(revs.App.tolist())

print(Ngapp)



NgappRev=revs[revs.App==Ngapp]

NgappRev.Translated_Review=NgappRev.Translated_Review.astype(str)

txt=NgappRev.Translated_Review.dropna().tolist()

NgappRevTxt=' '.join(txt)
nlp(NgappRevTxt)
!git clone https://github.com/huggingface/transformers.git 

!git checkout d6de6423

os.chdir("/kaggle/working/transformers")

!pip install -e ".[dev]"
import transformers

transformers.__version__
summarizer = pipeline("summarization")
def myRange(start,end,step):

    i = start

    while i < end:

        yield i

        i += step

    yield end
print(f'Summarising {len(GdappRevTxt)} words for {Gdapp}!')

pvs=0

for i in myRange(512,len(GdappRevTxt),512):

    print(summarizer(GdappRevTxt[pvs:i], min_length=5, max_length=75))

    pvs=i
print(f'Summarising {len(NgappRevTxt)} words for {Ngapp}!')

pvs=0

for i in myRange(512,len(NgappRevTxt),512):

    print(summarizer(NgappRevTxt[pvs:i], min_length=5, max_length=75))

    pvs=i
from sklearn.manifold import TSNE

featExt = pipeline("feature-extraction")
appGenre=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

revsGenre=revs.merge(appGenre, left_on='App', right_on='App')
examples=['Books & Reference','Photography','Action','Business','Sports']
tsne=TSNE(init='pca',learning_rate=10)

f, axes = plt.subplots(1, 5)



for i,gen in enumerate(examples):



    RanGen=revsGenre[revsGenre['Genres']==gen]

    ls=[]

    for app in RanGen['App'].unique():

        appRev=revs[revs['App']==app]

        appRev.Translated_Review=appRev['Translated_Review'].astype(str)

        txt=appRev.Translated_Review.dropna().tolist()

        appRevTxt=' '.join(txt)

        feat=featExt(appRevTxt)[0][0]

        ls.append(feat)

    y = tsne.fit_transform(np.asarray(ls))

    axes[i].scatter(y[:, 0], y[:, 1])

    axes[i].set_title(gen)



f.set_figwidth(15)

f.tight_layout()    
#https://github.com/jessevig/bertviz

!git clone https://github.com/jessevig/bertviz.git

import os

os.chdir('bertviz')

from transformers_neuron_view import BertModel, BertTokenizer

from neuron_view import show
def call_html():

  import IPython

  display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

        <script>

          requirejs.config({

            paths: {

              base: '/static/base',

              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min",

              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',

            },

          });

        </script>

        '''))
model_type = 'bert'

model_version = 'bert-base-uncased'

do_lower_case = True

model = BertModel.from_pretrained(model_version)

tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

sentence_a = "The cat sat on the mat"

sentence_b = "The cat lay on the rug"

call_html()

show(model, model_type, tokenizer, sentence_a, sentence_b)