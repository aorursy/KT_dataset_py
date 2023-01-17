# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os 

import re

import emoji

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

df=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df.head()
def missing_value_of_data(data):

    total = data.isnull().sum().sort_values(ascending=False)

    precent=round(total/data.shape[0]*100,2)

    return pd.concat([total,precent],axis=1,keys=['Total','Percent'])
missing_value_of_data(df)
df.dropna(inplace=True)
def count_values_in_columns(data,feature):

    total=data.loc[:,feature].value_counts()

    percentage = round(data.loc[:,feature].value_counts(normalize=True)*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
count_values_in_columns(df,'sentiment')
def unique_value_in_column(data,feature):

    unique_value=pd.Series(data.loc[:,feature].unique())

    return pd.concat([unique_value],axis=1,keys=['Unique_values'])
unique_value_in_column(df,'sentiment')
def duplicated_values_data(data):

    dup=[]

    columns=data.columns

    for i in columns:

        dup.append(sum(data[i].duplicated()))

    return pd.concat([pd.Series(columns),pd.Series(dup)],axis=1,keys=['Columns','Duplicate count'])    
duplicated_values_data(df)
df.describe()
def find_url(string):

    try:

        text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)

    except:

        text=[]

    return "".join(text)
sentence="Hello this is https://www.google.com"

find_url(sentence)
df['url'] = df['text'].apply(lambda x : find_url(x))
def find_emoji(text):

    emo_text=emoji.demojize(text)

    line=re.findall(r'\:(.*?)\:',emo_text)

    return line
sentence="I love ⚽ very much 😁"

find_emoji(sentence)
df['emoji'] = df['text'].apply(lambda x: find_emoji(x))
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
sentence="Its all about \U0001F600 face"

print(sentence)

remove_emoji(sentence)
df['text']=df['text'].apply(lambda x: remove_emoji(x))
def stop_word_fn(text):

    stop_words=set(stopwords.words('english'))

    word_tokens= word_tokenize(text)

    non_stop_word=[w for w in word_tokens if not w in stop_words ]

    stop_words = [w for w in word_tokens if w in stop_words]

    return stop_words
example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_word_fn(example_sent)
df['stop_words']=df['text'].apply(lambda x : stop_word_fn(x))
def jaccard(str1,str2):

    a=set(str1.lower().split())

    b=set(str2.lower().split())

    c=a.intersection(b)

    return float(len(c))/(len(a)+ len(b)+ len(c))

s1="Lets Play with the jaccard metrics whether its working"

s2="jaccard metrics"

s3 = "play working with metrics"

print(jaccard(s1,s2))

print(jaccard(s1,s3))

print(jaccard(s2,s3))
import numpy as np

import pandas as pd

import os

import re

import string

import matplotlib.pyplot as plt

import matplotlib_venn as venn

import seaborn as sns

from tqdm import tqdm

import spacy

import random

from spacy.util import compounding

from spacy.util import minibatch

from collections import defaultdict,Counter

from sklearn import preprocessing,model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk.corpus import stopwords

from nltk.util import ngrams

stop = set(stopwords.words('english'))

from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

from PIL import Image

import warnings

warnings.filterwarnings(action="ignore")

#plotly libraries

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

from plotly.subplots import make_subplots

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

from datetime import datetime as dt



train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

train.sample(6)
train.dropna()

train.shape
train['target'] = train['selected_text'].str.lower()
train.head()
train['target_url'] =train['target'].apply(lambda x : find_url(x))
def find_star(text):

    try:

        line=re.findall(r'[*]{2,5}',text)

    except:

        line=[]

    return len(line)

train['star']=train['target'].apply(lambda x:find_star(x))
def find_only_star(text):

    try:

        if len(text.split())==1:

            line=re.findall(r'[*]{2,5}',text)

            return len(line)

        else:

            return 0

    except:

        return 0

train['only_star']=train['target'].apply(lambda x:find_only_star(x))    
train['target']= np.where(train['only_star']==1,"abusive",train['target'])
def remove_link(string):

    try:

        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'," ",string)

    except:

        text=''

    return " ".join(text.split())

def remove_punct(text):

    try:

        line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]+'," ",text)

    except:

        line=''

    return " ".join(line.split())

train['target']=train['target'].apply(lambda x:remove_link(x))

train['target']=train['target'].apply(lambda x:remove_punct(x))

train['target_tweet_length']=train['target'].str.split().map(lambda x: len(x))
train['target_average_word_len']=train['target'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
full_data=pd.concat([train,test])

full_data['text']=full_data['text'].str.lower()

full_data.shape

full_data['text']=full_data['text'].apply(lambda x:remove_link(x))

full_data['text']=full_data['text'].apply(lambda x:remove_punct(x))
full_data.loc[full_data['text']=="",['text']]="nothing"
full_data['text_tweet_length']=full_data['text'].str.split().map(lambda x: len(x))

full_data['text_average_word_len']=full_data['text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
def corpus_sentiment_stop(data,feature,sentiment):

    corpus=create_corpus(data,feature,sentiment)

    dic=defaultdict(int)

    for word in corpus:

        if word in stop:

            dic[word]+=1

    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)

    x,y=zip(*top)

    return x,y
#Since we dont have length larger than 96

MAX_LEN = 96



# Pretrained model of roberta

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)



# Sentiment ID value is encoded from tokenizer

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')

ct=train.shape[0] #27481





input_ids=np.ones((ct,MAX_LEN),dtype="int32")          

attention_mask=np.zeros((ct,MAX_LEN),dtype="int32")    

token_type_ids=np.zeros((ct,MAX_LEN),dtype="int32")    

start_tokens=np.zeros((ct,MAX_LEN),dtype="int32")  

end_tokens=np.zeros((ct,MAX_LEN),dtype="int32")

for k in range(train.shape[0]):

#1 FIND OVERLAP

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    

    # idx - position where the selected text are placed. 

    idx = text1.find(text2)   # we get [12] position

    

    # all character position as 0 and then places 1 for selected text position  

    chars = np.zeros((len(text1))) 

    chars[idx:idx+len(text2)]=1    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] 

    

    #tokenize id of text 

    if text1[idx-1]==' ': chars[idx-1] = 1    

    enc = tokenizer.encode(text1)  #  [127, 3504, 16, 11902, 162]

        

#2. ID_OFFSETS - start and end index of text

    offsets = []

    idx=0

    for t in enc.ids:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))     #  [(0, 3), (3, 8), (8, 11), (11, 20), (20, 23)]

        idx += len(w) 

    

#3  START-END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b]) # number of characters in selected text - [0.0,0.0,0.0,9.0,3.0] - bullying me

        if sm>0: 

            toks.append(i)  # token position - selected text - [3, 4]

        

    s_tok = sentiment_id[train.loc[k,'sentiment']] # Encoded values by tokenizer

    

    #Formating input for roberta model

    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]   #[ 0   127  3504    16 11902   162     2     2  2430     2]

    attention_mask[k,:len(enc.ids)+5] = 1                                  # [1 1 1 1 1 1 1 1 1 1]

    

    if len(toks)>0:

        # this will produce (27481, 96) & (27481, 96) arrays where tokens are placed

        start_tokens[k,toks[0]+1] = 1

        end_tokens[k,toks[-1]+1] = 1 
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')



ct_test = test.shape[0]



# Initialize inputs

input_ids_t = np.ones((ct_test,MAX_LEN),dtype='int32')        # array with value 1 for shape (3534, 96)

attention_mask_t = np.zeros((ct_test,MAX_LEN),dtype='int32')  # array with value 0 for shape (3534, 96)

token_type_ids_t = np.zeros((ct_test,MAX_LEN),dtype='int32')  # array with value 0 for shape (3534, 96)



# Set Inputs attention 

for k in range(test.shape[0]):

        

#1. INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

     

    # Encoded value of tokenizer

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    

    #setting up of input ids - same as we did for train

    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask_t[k,:len(enc.ids)+5] = 1
def scheduler(epoch):

    return 3e-5 * 0.2**epoch

def build_model():

    

    # Initialize keras layers

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)



    # Fetching pretrained models 

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids,attention_mask=att,token_type_ids=tok)

    

    # Setting up layers

    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    # Initializing input,output for model.THis will be trained in next code

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    

    #Adam optimizer for stochastic gradient descent. if you are unware of it - https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)



    return model

    
n_splits=5 # Number of splits



# INitialize start and end token

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))



DISPLAY=1

for i in range(5):

    print('#'*40)

    print('### MODEL %i'%(i+1))

    print('#'*40)

    

    K.clear_session()

    model = build_model()

    # Pretrained model

    model.load_weights('../input/model4/v4-roberta-%i.h5'%i)



    print('Predicting Test...')

    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/n_splits

    preds_end += preds[1]/n_splits
all = []



for k in range(input_ids_t.shape[0]):

    # Argmax - Returns the indices of the maximum values along axis

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test.loc[k,'text']

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-1:b])

    all.append(st)

test['selected_text'] = all

submission=test[['textID','selected_text']]

submission.to_csv('submission.csv',index=False)

submission.head(5)
submission.shape