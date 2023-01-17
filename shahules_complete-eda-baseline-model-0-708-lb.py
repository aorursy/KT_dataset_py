!pip install textstat
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer,PorterStemmer

from collections import defaultdict,Counter

from wordcloud import WordCloud, STOPWORDS

from plotly.subplots import make_subplots

from nltk.tokenize import word_tokenize

import plotly.figure_factory as ff

from nltk.corpus import stopwords

from nltk.corpus import stopwords

import plotly.graph_objects as go

from textblob import TextBlob

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from nltk.util import ngrams

import plotly.offline as py

import plotly.express as px

from statistics import *

from plotly import tools

import seaborn as sns

from tqdm import tqdm

import pandas as pd

import numpy as np

import textstat

import string

import json

import nltk

import gc



py.init_notebook_mode(connected=True)

nltk.download('stopwords')

stop=set(stopwords.words('english'))

plt.style.use('seaborn')

train=pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

test=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

target=train['sentiment']
print('There are {} rows and {} cols in train set'.format(train.shape[0],train.shape[1]))

print('There are {} rows and {} cols in test set'.format(test.shape[0],test.shape[1]))
train.head(3)
import re

def basic_cleaning(text):

    text=re.sub(r'https?://www\.\S+\.com','',text)

    text=re.sub(r'[^A-Za-z|\s]','',text)

    return text



def clean(df):

    for col in ['text','selected_text']:

        df[col]=df[col].astype(str).apply(lambda x:basic_cleaning(x))

    return df



colors=['blue','green','red']

sent=train.sentiment.unique()
fig=make_subplots(1,2,subplot_titles=('Train set','Test set'))

x=train.sentiment.value_counts()

fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['blue','green','red'],name='train'),row=1,col=1)

x=test.sentiment.value_counts()

fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['blue','green','red'],name='test'),row=1,col=2)
df=pd.concat([train,test])

df['text']=df['text'].astype(str)

df['seleceted_text']=df['selected_text'].astype(str)
vals=[]

for i in range(0,3):

    x=df[df['sentiment']==sent[i]]['text'].str.len()

    vals.append(x)



fig = ff.create_distplot(vals, sent,show_hist=False)

fig.update_layout(title="Distribution of number of characters in tweets")

fig.show()
vals=[]

for i in range(0,3):

    x=df[df['sentiment']==sent[i]]['selected_text'].dropna().str.len()

    vals.append(x)



fig = ff.create_distplot(vals, sent)

fig.update_layout(title="Distribution of number of characters in selected text")

fig.show()


sent=df.sentiment.unique()

fig,ax= plt.subplots(1,3,figsize=(12,6))

for i in range(0,3):

    df[df['sentiment']==sent[i]]['text'].str.split().str.len().hist(ax=ax[i],color=colors[i])

    ax[i].set_title(sent[i])

fig.suptitle("Distribution of number of No: Words in Tweets", fontsize=14)


sent=train.sentiment.unique()

fig,ax= plt.subplots(1,3,figsize=(12,6))

for i in range(0,3):

    train[train['sentiment']==sent[i]]['selected_text'].str.split().str.len().hist(ax=ax[i],color=colors[i])

    ax[i].set_title(sent[i])

fig.suptitle("Distribution of number of No: Words in Selected text", fontsize=14)
def preprocess_news(df,stop=stop,n=1,col='text'):

    '''Function to preprocess and create corpus'''

    new_corpus=[]

    stem=PorterStemmer()

    lem=WordNetLemmatizer()

    for text in df[col]:

        words=[w for w in word_tokenize(text) if (w not in stop)]

       

        words=[lem.lemmatize(w) for w in words if(len(w)>n)]

     

        new_corpus.append(words)

        

    new_corpus=[word for l in new_corpus for word in l]

    return new_corpus



fig,ax=plt.subplots(1,3,figsize=(15,7))

for i in range(3):

    new=df[df['sentiment']==sent[i]]

    corpus_train=preprocess_news(new,{})

    

    dic=defaultdict(int)

    for word in corpus_train:

        if word  in stop:

            dic[word]+=1

            

    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    x,y=zip(*top)

    ax[i].bar(x,y,color=colors[i])

    ax[i].set_title(sent[i],color=colors[i])



fig.suptitle("Common stopwords in different sentiments")
df['punc']=df['text'].apply(lambda x : [c for c in x if c in string.punctuation])

fig,ax=plt.subplots(1,3,figsize=(10,6))

for i in range(3):

    new=df[df['sentiment']==sent[i]]['punc'].map(lambda x: len(x))

    sns.distplot(new,color=colors[i],ax=ax[i])

    ax[i].set_title(sent[i],color=colors[i])

    

fig.suptitle("Number of Punctuations in tweets") 
fig,ax=plt.subplots(1,3,figsize=(12,7))

for i in range(3):

    new=df[df['sentiment']==sent[i]]['text'].map(lambda x: len(set(x.split())))

    sns.distplot(new.values,ax=ax[i],color=colors[i])

    ax[i].set_title(sent[i])

fig.suptitle("Distribution of number of unique words")
fig,ax=plt.subplots(1,3,figsize=(12,7))

for i in range(3):

    new=df[df['sentiment']==sent[i]]['selected_text'].astype(str).map(lambda x: len(set(x.split())))

    sns.distplot(new.values,ax=ax[i],color=colors[i])

    ax[i].set_title(sent[i])

fig.suptitle("Distribution of number of unique words")
fig,ax=plt.subplots(1,3,figsize=(15,10))

for i in range(3):

    new=df[df['sentiment']==sent[i]]['punc']

    punc=[p for pun in new.values for p in pun]

    counter=Counter(punc).most_common(10)

    x,y=zip(*counter)

    ax[i].bar(x,y,color=colors[i])

    ax[i].set_title(sent[i],color=colors[i])

    

fig.suptitle("Punctuations in tweets")   

    

    
df=clean(df)


fig,ax=plt.subplots(1,3,figsize=(20,12))

for i in range(3):

    new=df[df['sentiment']==sent[i]]

    corpus_train=preprocess_news(new,n=3)

    counter=Counter(corpus_train)

    most=counter.most_common()

    x=[]

    y=[]

    for word,count in most[:20]:

        if (word not in stop) :

            x.append(word)

            y.append(count)

    sns.barplot(x=y,y=x,ax=ax[i],color=colors[i])

    ax[i].set_title(sent[i],color=colors[i])

fig.suptitle("Common words in tweet text")
fig,ax=plt.subplots(1,3,figsize=(20,12))

for i in range(3):

    new=df[df['sentiment']==sent[i]]   

    corpus=preprocess_news(new,n=3,col='selected_text')

    counter=Counter(corpus)

    most=counter.most_common()

    x=[]

    y=[]

    for word,count in most[:20]:

        if (word not in stop) :

            x.append(word)

            y.append(count)

    sns.barplot(x=y,y=x,ax=ax[i],color=colors[i])

    ax[i].set_title(sent[i],color=colors[i])

fig.suptitle("Common words in selected text")
def get_top_ngram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(n, n),stop_words=stop).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:20]
fig,ax=plt.subplots(1,2,figsize=(15,10))

for i in range(2):

    new=df[df['sentiment']==sent[i+1]]['selected_text']

    top_n_bigrams=get_top_ngram(new,2)[:20]

    x,y=map(list,zip(*top_n_bigrams))

    sns.barplot(x=y,y=x,ax=ax[i],color=colors[i+1])

    ax[i].set_title(sent[i+1])

    

fig.suptitle("Common bigrams in selected text")
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None,ax=None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=100,

        max_font_size=30, 

        scale=3,

        random_state=1 

        )

    

    wordcloud=wordcloud.generate(str(data))

    ax.imshow(wordcloud,interpolation='nearest')

    ax.axis('off')

    #plt.show()
fig,ax=plt.subplots(1,3,figsize=(20,12))

for i in range(3):

    new=df[df['sentiment']==sent[i]]['text']

    show_wordcloud(new,ax=ax[i])

    ax[i].set_title(sent[i],color=colors[i])
fig,ax=plt.subplots(1,3,figsize=(20,12))

for i in range(3):

    new=df[df['sentiment']==sent[i]]['selected_text'].dropna()

    show_wordcloud(new,ax=ax[i])

    ax[i].set_title(sent[i],color=colors[i])

    
#utility functions:

def plot_readability(a,b,c,title,bins=0.4,colors=colors):

    trace1 = ff.create_distplot([a,b,c],sent, bin_size=bins, colors=colors, show_rug=False)

    trace1['layout'].update(title=title)

    py.iplot(trace1, filename='Distplot')

    table_data= [["Statistical Measures","neu",'pos','neg'],

                ["Mean",mean(a),mean(b),mean(c)],

                ["Standard Deviation",pstdev(a),pstdev(b),pstdev(c)],

                ["Variance",pvariance(a),pvariance(b),pvariance(c)],

                ["Median",median(a),median(b),median(c)],

                ["Maximum value",max(a),max(b),max(c)],

                ["Minimum value",min(a),min(b),min(c)]]

    trace2 = ff.create_table(table_data)

    py.iplot(trace2, filename='Table')

tqdm.pandas()

fre_neu = np.array(df["text"][df["sentiment"] == sent[0]].progress_apply(textstat.flesch_reading_ease))

fre_pos = np.array(df["text"][df["sentiment"] == sent[1]].progress_apply(textstat.flesch_reading_ease))

fre_neg = np.array(df["text"][df["sentiment"] == sent[2]].progress_apply(textstat.flesch_reading_ease))



plot_readability(fre_neu,fre_pos,fre_neg,"Flesch Reading Ease",20)
def jaccard(str1, str2): 

    a = set(str(str1).lower().split()) 

    b = set(str(str2).lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))

def plot_jaccard(sentiment,ax):

    jacc=[]

    text=train[train['sentiment']==sentiment].dropna()['text'].values.tolist()

    selected=train[train['sentiment']==sentiment].dropna()['selected_text'].values.tolist()

    for i,k in zip(text,selected):

        jacc.append(jaccard(i,k))

    ax.hist(jacc,bins=10,color='blue',alpha=0.4)

    ax.set_title(sentiment)

    
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

plot_jaccard('positive',ax=ax1)

plot_jaccard('negative',ax2)

plot_jaccard('neutral',ax3)

fig.suptitle('jaccard similarity of text and selected text')
train['jaccard']=train.apply(lambda x : jaccard(x.text,x.selected_text),axis=1)

positive=train[(train['sentiment']=='positive') & (train['jaccard']>0.9)]
positive.head()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,7))

x=train[train['sentiment']=='positive']['text'].str.len()

y=train[(train['sentiment']=='positive')]['jaccard'].values.tolist()

ax1.scatter(x,y,color='green',alpha=.4)

ax1.set_xlabel('text length')

ax1.set_ylabel('jaccard similarity with selected text')

ax1.set_title("text length vs jaccard similarity")

x=train[train['sentiment']=='positive']['text'].apply(lambda x : len(x.split()))

ax2.scatter(x,y,color='green',alpha=.4)

ax2.set_xlabel('text length')

ax2.set_ylabel('jaccard similarity with selected text')

ax2.set_title("no: of words vs jaccard similarity")
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,7))

x=train[train['sentiment']=='negative']['text'].str.len()

y=train[(train['sentiment']=='negative')]['jaccard'].values.tolist()

ax1.scatter(x,y,color='red',alpha=.4)

ax1.set_xlabel('text length')

ax1.set_ylabel('jaccard similarity with selected text')

ax1.set_title("text length vs jaccard similarity")

x=train[train['sentiment']=='negative']['text'].apply(lambda x : len(x.split()))

ax2.scatter(x,y,color='red',alpha=.4)

ax2.set_xlabel('text length')

ax2.set_ylabel('jaccard similarity with selected text')

ax2.set_title("no: of words vs jaccard similarity")
def get_sent(text):

    testimonial = TextBlob(str(text))

    return testimonial.sentiment.polarity



plt.figure(figsize=(10,7))

train['polarity']=train['selected_text'].apply(lambda x : get_sent(x))

sns.boxplot(x='sentiment', y='polarity', data=train)

plt.gca().set_title('Sentiment vs Polarity of selected text')

plt.show()



import scipy

corr=[]

for i in sent:

    text_pos=train[train['sentiment']==i]['text'].astype(str).map(lambda x : len(x.split()))

    sel_pos=train[train['sentiment']==i]['selected_text'].astype(str).map(lambda x : len(x.split()))

    corr.append(scipy.stats.pearsonr(text_pos,sel_pos)[0])

plt.bar(sent,corr,color='blue',alpha=.7)

plt.gca().set_title("pearson corr between no: words in text and selected text")

plt.gca().set_ylabel("correlation")
from transformers import *

import tensorflow as tf

import tokenizers
MAXLEN=128

PATH = '../input/tf-roberta/'

tokenizer=tokenizers.ByteLevelBPETokenizer(vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True)



sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')



ct=train.shape[0]

input_ids=np.ones((ct,MAXLEN),dtype='int32')

attention_mask=np.zeros((ct,MAXLEN),dtype='int32')

token_type_ids=np.zeros((ct,MAXLEN),dtype='int32')

start_tokens=np.zeros((ct,MAXLEN),dtype='int32')

end_tokens=np.zeros((ct,MAXLEN),dtype='int32')





for k in tqdm(range(ct)):

    

    text1=" "+" ".join(train.loc[k,"text"].split())

    text2=" ".join(train.loc[k,"selected_text"].split())

    idx=text1.find(text2)

    chars=np.zeros(len(text1))

    chars[idx:idx+len(text2)]=1

    if (text1[idx-1]==" "):

        chars[idx-1]=1

    enc=tokenizer.encode(text1)

    

    offsets=enc.offsets

    

    toks=[]

    for i,(a,b) in enumerate(offsets):

        sm=np.sum(chars[a:b])

        if sm > 0:

            toks.append(i)

            

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask[k,:len(enc.ids)+5] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+1] = 1

        end_tokens[k,toks[-1]+1] = 1

    

    

test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')



ct = test.shape[0]

input_ids_t = np.ones((ct,MAXLEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAXLEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAXLEN),dtype='int32')



for k in tqdm(range(test.shape[0])):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask_t[k,:len(enc.ids)+5] = 1



def build_model():

    ids = tf.keras.layers.Input((MAXLEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAXLEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAXLEN,), dtype=tf.int32)



    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids,attention_mask=att,token_type_ids=tok)

    

    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x1 = tf.keras.layers.Conv1D(1,1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(1,1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)



    return model
model=build_model()
import os

models=os.listdir("../input/tfroberta5fold6epochs")

model_file_path="../input/tfroberta5fold6epochs/"
start_token=np.zeros((ct,MAXLEN),dtype='float64')

end_token=np.zeros((ct,MAXLEN),dtype='float64')



for model_path in models:

    model.load_weights(model_file_path+model_path)

    pred=model.predict([input_ids_t,attention_mask_t,token_type_ids_t])

    

    start_token+=pred[0]/len(models)

    end_token+=pred[1]/len(models)



start_token=np.argmax(start_token,axis=1)

end_token=np.argmax(end_token,axis=1)

selected_text=[]

for k in range(ct):

    

    a=start_token[k]

    b=end_token[k]

    if a<b:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        selected_text.append(tokenizer.decode(enc.ids[a-1:b]))

    else:

        if test.loc[k,'sentiment']=="neutral":

            selected_text.append(test.loc[k,'text'])

        else:

            text1 = " "+" ".join(test.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            selected_text.append(tokenizer.decode(enc.ids[b-1:a]))

            

    
test['selected_text']=selected_text

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.head(5)
del model

gc.collect()
cols=['textID','text','sentiment','selected_text']

train_df=train[cols].copy()

del train

test_df=test.copy()

del test

gc.collect()


import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from kaggle_datasets import KaggleDatasets

import transformers

from tqdm.notebook import tqdm

from tokenizers import BertWordPieceTokenizer

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional

from tensorflow.keras.models import Model

from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras.layers import concatenate

from tensorflow.compat.v1.keras.layers import CuDNNLSTM

import gc

import os
train_df=pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

test_df=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=128):

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  ## change it to commit



# Save the loaded tokenizer locally

save_path = '/kaggle/working/distilbert_base_uncased/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

tokenizer.save_pretrained(save_path)



# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowercase=True)

fast_tokenizer
x_train = fast_encode(train_df.text.astype(str), fast_tokenizer, maxlen=128)

x_test = fast_encode(test_df.text.astype(str),fast_tokenizer,maxlen=128)
transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
def create_targets(df):

    df['t_text'] = df['text'].apply(lambda x: tokenizer.tokenize(str(x)))

    df['t_selected_text'] = df['selected_text'].apply(lambda x: tokenizer.tokenize(str(x)))

    def func(row):

        x,y = row['t_text'],row['t_selected_text'][:]

        for offset in range(len(x)):

            d = dict(zip(x[offset:],y))

            #when k = v that means we found the offset

            check = [k==v for k,v in d.items()]

            if all(check)== True:

                break 

        return [0]*offset + [1]*len(y) + [0]* (len(x)-offset-len(y))

    df['targets'] = df.apply(func,axis=1)

    return df



train_df = create_targets(train_df)



print('MAX_SEQ_LENGTH_TEXT', max(train_df['t_text'].apply(len)))

print('MAX_TARGET_LENGTH',max(train_df['targets'].apply(len)))

MAX_TARGET_LEN=108
train_df['targets'] = train_df['targets'].apply(lambda x :x + [0] * (MAX_TARGET_LEN-len(x)))

targets=np.asarray(train_df['targets'].values.tolist())

lb=LabelEncoder()

sent_train=lb.fit_transform(train_df['sentiment'])

sent_test=lb.fit_transform(test_df['sentiment'])
def new_model(transformer_layer):

    

    inp = Input(shape=(128, ))

    inp2= Input(shape=(1,))

    

    embedding_matrix=transformer_layer.weights[0].numpy()



    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)



    x = CuDNNLSTM(150, return_sequences=True,name='lstm_layer',)(x)

    x = CuDNNLSTM(100, return_sequences=False,name='lstm_layer-2',)(x)

    

    y =Dense(10,activation='relu')(inp2)

    x= concatenate([x,y])

    

    x = Dense(MAX_TARGET_LEN,activation='sigmoid')(x)



    model = Model(inputs=[inp,inp2], outputs=x)



    model.compile(loss='binary_crossentropy',

                      optimizer='adam')





    #prinnt(model.summary())

    

    return model


model=new_model(transformer_layer)

history=model.fit([x_train,sent_train],targets,epochs=3)
predictions=model.predict([x_test,sent_test])




def convert_output(sub,predictions):

    preds=[]

    for i,row in enumerate(sub['text']):



        text,target=row.lower(),predictions[i].tolist()

        target=np.round(target).tolist()

        try:

            start,end=target.index(1),target[::-1].index(1)

            text_list=tokenizer.tokenize(text)

            text_list=text_list+((108-len(text_list))*['pad'])

            start_w,end_w=text_list[start],text_list[-end]

            start=text.find(start_w.replace("#",'',1))    ## remove # to match substring

            end=text.find(end_w.replace("#",''),start)

            #pred=' '.join([x for x in text_list[start:-end]])

            pred=text[start:end]

        except:

            pred=text

        

        preds.append(pred)

        

    return preds

prediction_text=convert_output(test_df,predictions)
len(prediction_text)