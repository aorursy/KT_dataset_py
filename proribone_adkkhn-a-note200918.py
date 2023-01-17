import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))|{'i','im','you','youre','they','theyre','he','hes','she','shes','we','our','us','were','arent',\
      'can','cant','could','couldnt','will','wont','would','wouldnt','should','shouldnt','may',\
       'dont','didnt','doesnt'}
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from wordcloud import WordCloud

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from gensim.models import word2vec
from janome.tokenizer import Tokenizer

pd.set_option("display.max_colwidth", 200)
tweet=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
tweet.head(3)
df=pd.concat([tweet,test],sort=False)
#Reference : https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
df['text']=df['text'].apply(lambda x:remove_URL(x))

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
df['text']=df['text'].apply(lambda x : remove_html(x))

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
df['text']=df['text'].apply(lambda x : remove_punct(x))


tweet=df[:len(tweet)]
test=df[len(tweet):]
tweet['target']=tweet['target'].apply(lambda x:int(x))
tweet.head()
disaster=tweet[tweet['target']==1]
not_disaster=tweet[tweet['target']==0]
disaster.head(10)
def CountWord(S):
    cntD,cntN=0,0
    for s in disaster['text']:
        isok=False
        for c in s.split():
            if c.lower()==S:
                isok=True
                break
        if isok:
            cntD+=1
            
    for s in not_disaster['text']:
        isok=False
        for c in s.split():
            if c.lower()==S:
                isok=True
                break
        if isok:
            cntN+=1
    return cntD,cntN
def words_list(df):
    words=[]
    for x in tweet[tweet['target']==df]['text'].str.split():
        for s in x:
            if not s.lower() in stop:
                words.append(s.lower())
    return words
dwords=words_list(1)
d=defaultdict(int)
for word in dwords:
    d[word]+=1
top=sorted(d.items(),key=lambda x:x[1],reverse=True)[:30]
fig = plt.figure(figsize=(8.0, 8.0))
x,y=zip(*top)
plt.barh(x,y)
plt.title('Common words in disaster tweets')
d=defaultdict(int)
for word in dwords:
    d[word]+=1
top=sorted(d.items(),key=lambda x:x[1],reverse=True)[:300]

LI=[]
for w,c in top:
    cntD,cntN=CountWord(w)
    LI.append((w,cntD,cntN,(cntD/(cntD+cntN))))
LI.sort(key=lambda x:x[3],reverse=True)

for i,x in enumerate(LI):
    if i>=30:
        break
    w,a,b,p=x
    print(w)
    print('disaster:{:.5f}%'.format(p*100))
    print('non-disaster:{:.5f}%'.format(100-p*100))
    print('{0}/{1}'.format(a,a+b))
    print()
nwords=words_list(0)
n_d=defaultdict(int)
for word in nwords:
    n_d[word]+=1
top=sorted(n_d.items(),key=lambda x:x[1],reverse=True)[:30]
fig=plt.figure(figsize=(8.0, 8.0))
x,y=zip(*top)
plt.barh(x,y)
plt.title('Common words in non-disaster tweets')
d=defaultdict(int)
for word in nwords:
    d[word]+=1
top=sorted(d.items(),key=lambda x:x[1],reverse=True)[:300]
LI=[]
for w,c in top:
    cntD,cntN=CountWord(w)
    LI.append((w,cntD,cntN,(cntD/(cntD+cntN))))
LI.sort(key=lambda x:x[3])

for i,x in enumerate(LI):
    if i>=30:
        break
    w,a,b,p=x
    print(w)
    print('disaster:{:.5f}%'.format(p*100))
    print('non-disaster:{:.5f}%'.format(100-p*100))
    print('{0}/{1}'.format(a,a+b))
    print()
#target==dfでWを含むツイートを全て表示

def view(W,df):
    for s in tweet[tweet['target']==df]['text']:
        for c in s.split():
            if c.lower()==W:
                print(s)
                break
view('happy',1)
view('apocalypse',0)
def create_wordcloud(text,stop,df):
    wordcloud=WordCloud(background_color="white" if df==0 else "black",width=900,height=500,\
                       stopwords=stop).generate(text)
    plt.figure(figsize=(15,12))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
create_wordcloud(' '.join(dwords),stop,1)
create_wordcloud(' '.join(nwords),stop,0)
Disaster_Corpus=[]
Non_disaster_Corpus=[]
for s in tweet[tweet['target']==1]['text'].str.split():
    arr=[]
    for c in s:
        arr.append(c.lower())
    Disaster_Corpus.append(arr.copy())
for s in tweet[tweet['target']==0]['text'].str.split():
    arr=[]
    for c in s:
        arr.append(c.lower())
    Non_disaster_Corpus.append(arr.copy())
Disaster_Corpus[:10]
Non_disaster_Corpus[:10]
Corpus=Disaster_Corpus+Non_disaster_Corpus
model=word2vec.Word2Vec(Corpus,window=10,min_count=3)
model.wv.doesnt_match(["murder","terrorism","youtube"])