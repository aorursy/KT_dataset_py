from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

from datetime import datetime

from pytz import timezone

datetime.now(timezone('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M:%S')



def refer_args(x):

    if type(x) == 'method':

        print(*x.__code__.co_varnames.split(), sep='\n')

    else:

        print(*[x for x in dir(x) if not x.startswith('__')], sep='\n')
from collections import Counter, defaultdict

import os

from operator import itemgetter

import re

import string



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from tqdm import tqdm

from nltk.corpus import stopwords

from nltk.util import ngrams

from nltk.tokenize import word_tokenize

import gensim

from gensim.models import word2vec

from wordcloud import WordCloud

from janome.tokenizer import Tokenizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from keras.optimizers import Adam

 

plt.style.use('ggplot')

stop = set(stopwords.words('english')) | {

            'i','im','you','youre','they','theyre','he','hes','she','shes','we','our','us','were','arent',\

            'can','cant','could','couldnt','will','wont','would','wouldnt','should','shouldnt','may',\

            'dont','didnt','doesnt'}



pd.set_option('display.max_colwidth', 200)
from gensim.models.doc2vec import Doc2Vec,TaggedDocument

Pretrained_Model=Doc2Vec.load('../input/pretrained-0923-2249/Pretrained_2249.model')
tweet=pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')
df=pd.concat([tweet,test],sort=False)
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)

df['text']=df['text'].apply(lambda x: remove_URL(x))



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

df['text']=df['text'].apply(lambda x: remove_html(x))



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

df['text']=df['text'].apply(lambda x: remove_emoji(x))



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)

df['text']=df['text'].apply(lambda x: remove_punct(x))



def string_lower(text):

    return text.lower()

df['text']=df['text'].apply(lambda x: string_lower(x))
tweet=df[:len(tweet)]

test=df[len(tweet):]
words1=tweet['text'][5].split()

words2=tweet['text'][3].split()

' '.join(words1)

' '.join(words2)
Pretrained_Model.docvecs.similarity_unseen_docs(Pretrained_Model,words1,words2,alpha=1,min_alpha=0.0001,steps=5)
newvecs=[Pretrained_Model.infer_vector(tweet['text'][i].split()) for i in range(len(tweet))]

train=pd.DataFrame(data=newvecs)

train.head()
Y_train=tweet['target'].apply(lambda x:int(x))

Y_train.head()
tweet.head()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam
model=Sequential()

model.add(Dense(256,activation="relu"))

model.add(Dense(128,activation="relu"))

model.add(Dense(1,activation="sigmoid"))

optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

model.fit(train,Y_train,epochs=200,validation_split=0.2)
model.summary()
Test_newvecs=[Pretrained_Model.infer_vector(test['text'][i].split()) for i in range(len(test))]
TEST=pd.DataFrame(data=Test_newvecs)
TEST.head()
predict=model.predict(TEST)
predict=np.round(predict).astype(int).reshape(3263)
sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sub['target']=predict
sub.to_csv('submission.csv',index=False)
sub.head()