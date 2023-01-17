%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import sqlite3

import pandas as pd

import numpy as np

import nltk

import math

import scipy.stats

import string

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



import pickle



from tqdm import tqdm

import os



import plotly.graph_objs as go

from collections import Counter

from wordcloud import WordCloud
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_data  = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print(train_data.shape)

print(test_data.shape)
train_data.columns
import datetime



x = datetime.datetime.now()

print(x)



minute_now = int(x.strftime("%M"))



print(minute_now)



if(minute_now<44):

    test_data.to_csv('submission.csv',index=False)

    train_data = []

    test_data = []
print(train_data.groupby(train_data['sentiment']).size())

class_label = train_data.sentiment.value_counts()

sns.barplot(class_label.index, class_label)

plt.gca().set_ylabel('samples')
## Checking for the number of nan values in text and selected text column



print("No of nan values in text column = ",train_data['text'].isna().sum())

print("No of nan values in selected_text column = ",train_data['selected_text'].isna().sum())



print("\nId of null text column = ", train_data[train_data['text'].isna()]['textID'])

print("Id of null selected_text column = ", train_data[train_data['selected_text'].isna()]['textID'])



train_data = train_data.fillna("")

test_data = test_data.fillna("")



print(train_data.shape)

print(test_data.shape)
# Analysis of length





print("Length of shortest text = ", min(train_data['text'].str.len()))

print("Length of Longest text = ", max(train_data['text'].str.len()))





fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16,8))



#For Positive Review

sns.distplot(train_data[train_data['sentiment']=='positive']['text'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'darkblue', 

             ax = ax1,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax1.set_title('Positive Reviews')

ax1.set_xlabel('Text_Length')

ax1.set_ylabel('Density')



#For Negative Review

sns.distplot(train_data[train_data['sentiment']=='negative']['text'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'red', 

             ax = ax2,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax2.set_title('Negative Reviews')

ax2.set_xlabel('Text_Length')

ax2.set_ylabel('Density')



#For Neutral Review

sns.distplot(train_data[train_data['sentiment']=='neutral']['text'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'brown', 

             ax = ax3,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax3.set_title('Neutral Reviews')

ax3.set_xlabel('Text_Length')

ax3.set_ylabel('Density')

# Central Tendency Analysis





#For Positive Review

mn = np.mean(train_data[train_data['sentiment']=='positive']['text'].str.len())

md = np.median(train_data[train_data['sentiment']=='positive']['text'].str.len())

print('Mean length of positive review is ', mn)

print('Median length of positive review is ', md)



#For Negative Review

mn = np.mean(train_data[train_data['sentiment']=='negative']['text'].str.len())

md = np.median(train_data[train_data['sentiment']=='negative']['text'].str.len())

print('\nMean length of Negative review is ', mn)

print('Median length of Negative review is ', md)



#For Neutral Review

mn = np.mean(train_data[train_data['sentiment']=='neutral']['text'].str.len())

md = np.nanmedian(train_data[train_data['sentiment']=='neutral']['text'].str.len())

print('\nMean length of Neutral review is ', mn)

print('Median length of Neutral review is ', md)
# Stopwords analysis of text feature



stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]



arr = []



for i in train_data['text']:

    words = i.split()

    cnt = 0

    for i in words:

        if(i in stopwords):

            cnt+=1

    if(cnt!=0):

        k = round((float(cnt)/len(words))*100.0, 0)

    else:

        k = 0

    arr.append(k)

    

print("There are on average {}% stopwords in one text feature\n".format(round(np.mean(arr), 0)))

# Word Cloud Analysis Helper functions



def create_corpus(target, feature):

    corpus=[]

    

    for x in train_data[train_data['sentiment']==target][feature].str.split():

        for i in x:

            corpus.append(i)

    return corpus



def plot_wordcloud(corpus):

    plt.figure(figsize=(12,8))

    word_cloud = WordCloud(

                          background_color='black',

                          max_font_size = 80

                         ).generate(" ".join(corpus[:50]))

    plt.imshow(word_cloud)

    plt.axis('off')

    plt.show()
# Generating the wordcloud for positive sentiment

plot_wordcloud(create_corpus("positive","text"))
# Generating the wordcloud for negative sentiment

plot_wordcloud(create_corpus("negative","text"))
# Generating the wordcloud for neutral sentiment

plot_wordcloud(create_corpus("neutral","text"))
#Analysis of length of selected_text feature



print("Length of shortest selected_text = ", min(train_data['selected_text'].str.len()))

print("Length of Longest selected_text = ", max(train_data['selected_text'].str.len()))





fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16,8))



#For Positive Review

sns.distplot(train_data[train_data['sentiment']=='positive']['selected_text'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'darkblue', 

             ax = ax1,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax1.set_title('Positive Reviews')

ax1.set_xlabel('Text Length')

ax1.set_ylabel('Density')



#For Negative Review

sns.distplot(train_data[train_data['sentiment']=='negative']['selected_text'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'red', 

             ax = ax2,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax2.set_title('Negative Reviews')

ax2.set_xlabel('Text Length')

ax2.set_ylabel('Density')



#For Neutral Review

sns.distplot(train_data[train_data['sentiment']=='neutral']['selected_text'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'brown', 

             ax = ax3,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax3.set_title('Neutral Reviews')

ax3.set_xlabel('Text Length')

ax3.set_ylabel('Density')
# Stopwords analysis



arr = []



for i in train_data['selected_text']:

    words = i.split()

    cnt = 0

    for i in words:

        if(i in stopwords):

            cnt+=1

    if(cnt!=0):

        k = round((float(cnt)/len(words))*100.0, 0)

    else:

        k = 0

    arr.append(k)

    

print("There are on average {}% stopwords in one selected_text\n".format(round(np.mean(arr), 0)))

#Central Tendency of selected_text



#For Positive Review

mn = np.mean(train_data[train_data['sentiment']=='positive']['selected_text'].str.len())

md = np.median(train_data[train_data['sentiment']=='positive']['selected_text'].str.len())

print('Mean length of selected_text in positive review is ', mn)

print('Median length of selected_text in positive review is ', md)



#For Negative Review

mn = np.mean(train_data[train_data['sentiment']=='negative']['selected_text'].str.len())

md = np.median(train_data[train_data['sentiment']=='negative']['selected_text'].str.len())

print('\nMean length of selected_text in Negative review is ', mn)

print('Median length of selected_text in Negative review is ', md)



#For Neutral Review

mn = np.mean(train_data[train_data['sentiment']=='neutral']['selected_text'].str.len())

md = np.nanmedian(train_data[train_data['sentiment']=='neutral']['selected_text'].str.len())

print('\nMean length of selected_text in Neutral review is ', mn)

print('Median length of selected_text in Neutral review is ', md)
# Word Cloud of selected_text analysis

# Generating the wordcloud for positive sentiment

plot_wordcloud(create_corpus("positive","selected_text"))
# Generating the wordcloud for positive sentiment

plot_wordcloud(create_corpus("negative","selected_text"))
# Generating the wordcloud for positive sentiment

plot_wordcloud(create_corpus("neutral","selected_text"))
# How much selected_text and text feature matches



x = []

y = []



same_cnt = 0

arr = []

sentences = []

length = train_data.shape[0]



for index, row in train_data.iterrows():

    first = row['text'].split()

    d = {}

    for j in first:

        if(d.get(j)):

            d[j] = d[j]+1

        else:

            d[j] = 1



    cnt = 0

    scd = row['selected_text'].split()

    for j in scd:

        if(d.get(j)!=None and d[j]>0):

            cnt+=1

            d[j]-=1;

    if(cnt==len(scd)):

        same_cnt+=1

    else:

        arr.append(round((cnt/float(len(scd)))*100.0,0))

        sentences.append([row['text'], row['selected_text']])

    x.append(len(scd))

    y.append(cnt)



ss = round((same_cnt/float(train_data.shape[0]))*100.0,0)

mn = round(np.mean(arr), 0)

print("Almost {0}% of selected_text are strict subset of text ".format(ss))

print("\nOut of {0}% remaining selected_text there is almost {1}% average match of selected_text and text column".format(100-ss, mn))

# Jaccard simlarity analysis

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
length = train_data.shape[0]

pos_score = []

neg_score = []

neu_score = []



for index, row in train_data.iterrows():

    sent =row['sentiment']

    

    if(sent == 'positive'):

        pos_score.append(jaccard(row['text'], row['selected_text']))

    if(sent == 'negative'):

        neg_score.append(jaccard(row['text'], row['selected_text']))

    if(sent == 'neutral'):

        neu_score.append(jaccard(row['text'], row['selected_text']))



print("Mean jaccard score between text and selected_text for positive sentiment is {0}".format(np.mean(pos_score)))

print("Mean jaccard score between text and selected_text for negative sentiment is {0}".format(np.mean(neg_score)))

print("Mean jaccard score between text and selected_text for neutral sentiment is {0}".format(np.mean(neu_score)))

# Thanks to Raenish David for this excellent code of the plot

# Jaccard score for text vs selected_text for positive sentiment 



x = train_data[train_data['sentiment']=='positive']['text'].str.len()

y = pos_score

fig = go.Figure()

fig.add_trace(go.Histogram2dContour(

        x = x,

        y = y,

        colorscale = 'gray',

        reversescale = True,

        xaxis = 'x',

        yaxis = 'y'

    ))

fig.add_trace(go.Scatter(

        x = x,

        y = y,

        xaxis = 'x',

        yaxis = 'y',

        mode = 'markers',

        marker = dict(

            color = 'green',  #'rgba(0,0,0,0.3)',

            size = 3

        )

    ))

fig.add_trace(go.Histogram(

        y = y,

        xaxis = 'x2',

        marker = dict(

            color = 'rgba(0,0,0,1)'

        )

    ))

fig.add_trace(go.Histogram(

        x = x,

        yaxis = 'y2',

        marker = dict(

            color = 'rgba(0,0,0,1)'

        )

    ))



fig.update_layout(

    autosize = False,

    xaxis = dict(

        zeroline = False,

        domain = [0,0.85],

        showgrid = False

    ),

    yaxis = dict(

        zeroline = False,

        domain = [0,0.85],

        showgrid = False

    ),

    xaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    yaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    height = 600,

    width = 600,

    bargap = 0,

    hovermode = 'closest',

    showlegend = False,

    title_text="Postive Jaccard - Text vs Selected Text ",title_x=0.5

)



fig.show()
# Jaccard score for text vs selected_text for negative sentiment 



x = train_data[train_data['sentiment']=='negative']['text'].str.len()

y = neg_score

fig = go.Figure()

fig.add_trace(go.Histogram2dContour(

        x = x,

        y = y,

        colorscale = 'gray',

        reversescale = True,

        xaxis = 'x',

        yaxis = 'y'

    ))

fig.add_trace(go.Scatter(

        x = x,

        y = y,

        xaxis = 'x',

        yaxis = 'y',

        mode = 'markers',

        marker = dict(

            color = 'red',  #'rgba(0,0,0,0.3)',

            size = 3

        )

    ))

fig.add_trace(go.Histogram(

        y = y,

        xaxis = 'x2',

        marker = dict(

            color = 'rgba(0,0,0,1)'

        )

    ))

fig.add_trace(go.Histogram(

        x = x,

        yaxis = 'y2',

        marker = dict(

            color = 'rgba(0,0,0,1)'

        )

    ))



fig.update_layout(

    autosize = False,

    xaxis = dict(

        zeroline = False,

        domain = [0,0.85],

        showgrid = False

    ),

    yaxis = dict(

        zeroline = False,

        domain = [0,0.85],

        showgrid = False

    ),

    xaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    yaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    height = 600,

    width = 600,

    bargap = 0,

    hovermode = 'closest',

    showlegend = False,

    title_text="Negative Jaccard - Text vs Selected Text ",title_x=0.5

)



fig.show()
# Jaccard score for text vs selected_text for neutral sentiment 



x = train_data[train_data['sentiment']=='neutral']['text'].str.len()

y = neu_score

fig = go.Figure()

fig.add_trace(go.Histogram2dContour(

        x = x,

        y = y,

        colorscale = 'gray',

        reversescale = True,

        xaxis = 'x',

        yaxis = 'y'

    ))

fig.add_trace(go.Scatter(

        x = x,

        y = y,

        xaxis = 'x',

        yaxis = 'y',

        mode = 'markers',

        marker = dict(

            color = 'blue',  #'rgba(0,0,0,0.3)',

            size = 3

        )

    ))

fig.add_trace(go.Histogram(

        y = y,

        xaxis = 'x2',

        marker = dict(

            color = 'rgba(0,0,0,1)'

        )

    ))

fig.add_trace(go.Histogram(

        x = x,

        yaxis = 'y2',

        marker = dict(

            color = 'rgba(0,0,0,1)'

        )

    ))



fig.update_layout(

    autosize = False,

    xaxis = dict(

        zeroline = False,

        domain = [0,0.85],

        showgrid = False

    ),

    yaxis = dict(

        zeroline = False,

        domain = [0,0.85],

        showgrid = False

    ),

    xaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    yaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    height = 600,

    width = 600,

    bargap = 0,

    hovermode = 'closest',

    showlegend = False,

    title_text="Neutral Jaccard - Text vs Selected Text ",title_x=0.5

)



fig.show()
import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

print('TF version',tf.__version__)
train = train_data

test  = test_data

MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

EPOCHS = 3 # originally 3

BATCH_SIZE = 32 # originally 32

PAD_ID = 1

SEED = 88888

LABEL_SMOOTHING = 0.1

tf.random.set_seed(SEED)

np.random.seed(SEED)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(train.shape[0]):

    

    # FIND OVERLAP

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    idx = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

        

    # ID_OFFSETS

    offsets = []; idx=0

    for t in enc.ids:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    # START END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask[k,:len(enc.ids)+3] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+2] = 1

        end_tokens[k,toks[-1]+2] = 1
ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask_t[k,:len(enc.ids)+3] = 1
import pickle



def save_weights(model, dst_fn):

    weights = model.get_weights()

    with open(dst_fn, 'wb') as f:

        pickle.dump(weights, f)





def load_weights(model, weight_fn):

    with open(weight_fn, 'rb') as f:

        weights = pickle.load(f)

    model.set_weights(weights)

    return model



def loss_fn(y_true, y_pred):

    # adjust the targets for sequence bucketing

    ll = tf.shape(y_pred)[1]

    y_true = y_true[:, :ll]

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,

        from_logits=False, label_smoothing=LABEL_SMOOTHING)

    loss = tf.reduce_mean(loss)

    return loss





def build_model():

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)



    lens = MAX_LEN - tf.reduce_sum(padding, -1)

    max_len = tf.reduce_max(lens)

    ids_ = ids[:, :max_len]

    att_ = att[:, :max_len]

    tok_ = tok[:, :max_len]



    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)

    

    x1 = tf.keras.layers.Dropout(0.1)(x[0])

    x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)

    x1 = tf.keras.layers.ReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)

    x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 

    model.compile(loss=loss_fn, optimizer=optimizer)

    

    # this is required as `model.predict` needs a fixed size!

    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])

    return model, padded_model
jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    K.clear_session()

    model, padded_model = build_model()

        

    #sv = tf.keras.callbacks.ModelCheckpoint(

    #    '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,

    #    save_weights_only=True, mode='auto', save_freq='epoch')

    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]

    targetT = [start_tokens[idxT,], end_tokens[idxT,]]

    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]

    targetV = [start_tokens[idxV,], end_tokens[idxV,]]

    # sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = '%s-roberta-%i.h5'%(VER,fold)

    for epoch in range(1, EPOCHS + 1):

        # sort and shuffle: We add random numbers to not have the same order in each epoch

        shuffleT = np.int32(sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))

        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch

        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)

        batch_inds = np.random.permutation(num_batches)

        shuffleT_ = []

        for batch_ind in batch_inds:

            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])

        shuffleT = np.concatenate(shuffleT_)

        # reorder the input data

        inpT = [arr[shuffleT] for arr in inpT]

        targetT = [arr[shuffleT] for arr in targetT]

        model.fit(inpT, targetT, 

            epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY, callbacks=[],

            validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`

        save_weights(model, weight_fn)



    print('Loading model...')

    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))

    load_weights(model, weight_fn)



    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

    

    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        if a>b: 

            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[a-2:b-1])

        all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))
print(jac) # Jaccard CVs
all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test.loc[k,'text']

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-2:b-1])

    all.append(st)
test['selected_text'] = all

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)