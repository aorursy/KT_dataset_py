# basic

# =====

import numpy as np 

import pandas as pd



# Visualizations

# ==============

import matplotlib.pyplot as plt

import seaborn as sns

# import plotly_express as px





from sklearn.model_selection import StratifiedKFold



from transformers import *

import tokenizers



import tensorflow as tf

import tensorflow.keras.backend as K
# machine learning

# ================

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# from sklearn.model_selection import cross_val_score, train_test_split

# from sklearn.linear_model import RidgeClassifier, LogisticRegression

# from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.svm import SVC

# from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import accuracy_score, confusion_matrix



# nlp

# ===

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# from nltk.corpus import stopwords

# from wordcloud import WordCloud, STOPWORDS 

# import nltk

# import spacy



# deep learning

# =============

# import tensorflow as tf

# from tensorflow.keras.preprocessing.text import Tokenizer

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# palette

# =======

tw_pal = ['#1DCAFF', '#292F33', '#E0245E', '#E1E8ED', '#CCD6DD', '#E1E8ED']

tw_pal_1 = ['#1DCAFF', '#E0245E']



sns.set_style("whitegrid")
# list files

# ! ls ../input/tweet-sentiment-extraction
# load data

train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
# random rows from training dataset

train.sample(5)
# random rows from test dataset

test.sample(5)
# random rows from sample submission files

test.sample(5)
# shape of the dataset

print(train.shape)

print(test.shape)
# info of the dataset

print(train.info())

print("")

print(test.info())
# describe the dataset

print(train.describe())

print("")

print(test.describe())
# number of missing values

print(train.isna().sum())

print("")

print(test.isna().sum())

print("")

print(sample.isna().sum())
# row with missing values

train[train.isna().any(axis=1)]
# droping rows with missing value

train = train.dropna()



# total missing values after dropping rows with missing values

print(train.isna().sum().sum())
# no. of unique values()

print(train.nunique())

print('')

print(test.nunique())

print('')

print(sample.nunique())
# figure properties

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

fig.subplots_adjust(wspace=0.4)



# train dataset distribution

sns.countplot(x='sentiment', data=train, order=['positive', 'neutral', 'negative'], palette=tw_pal, ax=axes[0])

axes[0].set_title('Train dataset')

axes[0].set_xlabel('')

axes[0].set_ylabel('')



# test dataset distribution

sns.countplot(x='sentiment', data=test, order=['positive', 'neutral', 'negative'], palette=tw_pal, ax=axes[1])

axes[1].set_title('Test dataset')

axes[1].set_xlabel('')

axes[1].set_ylabel('')



plt.show()
# table

temp = train.groupby(['selected_text', 'sentiment']).count()

temp = temp.reset_index().sort_values('textID', ascending=False)

# temp.head()



# plot 

plt.figure(figsize=(10, 6))

ax = sns.barplot(data=temp.head(20), x='textID', y='selected_text', hue='sentiment', dodge=False, palette=tw_pal_1)

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Top 20 selected text')

plt.show()
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

fig.subplots_adjust(wspace=0.4)



sns.barplot(data=temp[temp['sentiment']=='positive'].head(6), x='textID', y='selected_text', 

            dodge=False, color=tw_pal[0], ax=axes[0])

axes[0].set_title('Top positive selected text')

axes[0].set_ylabel('')

axes[0].set_xlabel('')



sns.barplot(data=temp[temp['sentiment']=='neutral'].head(6), x='textID', y='selected_text', 

            dodge=False, color=tw_pal[1], ax=axes[1])

axes[1].set_title('Top neutral selected text')

axes[1].set_ylabel('')

axes[1].set_xlabel('')



sns.barplot(data=temp[temp['sentiment']=='negative'].head(6), x='textID', y='selected_text', 

            dodge=False, color=tw_pal[2], ax=axes[2])

axes[2].set_title('Top negative selected text')

axes[2].set_ylabel('')

axes[2].set_xlabel('')



plt.show()
# utility function

# =================



def plot_bar(col, col_names):

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    fig.subplots_adjust(wspace=0.4)

    for ind, val in enumerate(col):

        sns.barplot(x='sentiment', y=val, palette=tw_pal, data=train, ax=ax[ind], order=['positive', 'neutral', 'negative'])

        ax[ind].set_title(col_names[ind])

        ax[ind].set_ylabel('')

        ax[ind].set_xlabel('')

    plt.plot()
# no. of characters

# =================



train['no_chars_text'] = train['text'].apply(len)

train['no_chars_sel_text'] = train['selected_text'].apply(len)



col = ['no_chars_text', 'no_chars_sel_text']

col_names = ['Mean no. of chars in text', 'Mean no. of chars in selected text']



plot_bar(col, col_names)
# no. of words

# ============



train['no_words'] = train['text'].str.split().apply(len)

train['no_words_sel_text'] = train['selected_text'].str.split().apply(len)



col = ['no_words', 'no_words_sel_text']

col_names = ['Mean no. of words in text', 'Mean no. of words in selected text']



plot_bar(col, col_names)
# no. of sentences

# ================



train['no_sent'] = train['text'].str.split('.').apply(len)

train['no_sent_sel_text'] = train['selected_text'].str.split('.').apply(len)



col = ['no_sent', 'no_sent_sel_text']

col_names = ['Mean no. of sentances in text', 'Mean no. of sentances in selected text']



plot_bar(col, col_names)
# utility function

# =================



def hash_count(tweet):

    w = tweet.split()

    return len([word for word in w if word.startswith('#')])



def mention_count(tweet):

    w = tweet.split()

    return len([word for word in w if word.startswith('@')])



def web_add(tweet):

    w = tweet.split()

    return len([word for word in w if word.startswith('http')])
# no. of hashtags

# ===============



train['no_hashtags'] = train['text'].apply(hash_count)

train['no_hashtags_sel_text'] = train['selected_text'].apply(hash_count)



col = ['no_hashtags', 'no_hashtags_sel_text']

col_names = ['Mean no. of hashtags in text', 'Mean no. of hashtags in selected text']



plot_bar(col, col_names)
# no. of mentions

# ===============



train['no_mentions'] = train['text'].apply(mention_count)

train['no_mentions_sel_text'] = train['selected_text'].apply(mention_count)



col = ['no_mentions', 'no_mentions_sel_text']

col_names = ['Mean no. of mentions in text', 'Mean no. of mentions in selected text']



plot_bar(col, col_names)
# no. of web address

# ==================



train['no_web_add'] = train['text'].apply(hash_count)

train['no_web_add_sel_text'] = train['selected_text'].apply(hash_count)



col = ['no_web_add', 'no_web_add_sel_text']

col_names = ['Mean no. of web address in text', 'Mean no. of web address in selected text']



plot_bar(col, col_names)
import pandas as pd, numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

print('TF version',tf.__version__)
def read_train():

    train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

    train['text']=train['text'].astype(str)

    train['selected_text']=train['selected_text'].astype(str)

    return train



def read_test():

    test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

    test['text']=test['text'].astype(str)

    return test



def read_submission():

    test=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

    return test

    

train_df = read_train()

test_df = read_test()

submission_df = read_submission()
# def jaccard_improve(str1, str2): 

#     str1 = str1.lower()

#     str2 = str2.lower()    

    

#     index = str1.find(str2) 

#     text1 = str1[:index]

#     #print(text1)

    

#     text2 = str1[index:].replace(str2, '')

#     words1 = text1.split()

#     words2 = text2.split()

#     #print(words1[-3:])



#     if len(words1) > len(words2):

#         words1 = words1[-3:]

#         mod_text = " ".join(words1) + " " + str2

#     else:

#         words2 = words2[0:2]

#         mod_text = str2 + " " + " ".join(words2)

    

#     return mod_text 
def jaccard(str1, str2): 

    a = set(str(str1).lower().split()) 

    b = set(str(str2).lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
# print(len(train_df))
# train_df['selected_text_mod'] = train_df['selected_text']

# train_df['mod'] = 0



# train_df.head()
MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
ct = train_df.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(train_df.shape[0]):

    

    # FIND OVERLAP

    text1 = " "+" ".join(train_df.loc[k,'text'].split())

    text2 = " ".join(train_df.loc[k,'selected_text'].split())

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

        

    s_tok = sentiment_id[train_df.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask[k,:len(enc.ids)+5] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+1] = 1

        end_tokens[k,toks[-1]+1] = 1
# for k in range(train_df.shape[0]):

    

#     # FIND OVERLAP

#     text1 = " "+" ".join(train_df.loc[k,'text'].split())

#     text2 = " ".join(train_df.loc[k,'selected_text'].split())

#     idx = text1.find(text2)

#     chars = np.zeros((len(text1)))

#     chars[idx:idx+len(text2)]=1

#     if text1[idx-1]==' ': chars[idx-1] = 1 

#     enc = tokenizer.encode(text1) 

        

#     # ID_OFFSETS

#     offsets = []; idx=0

#     for t in enc.ids:

#         w = tokenizer.decode([t])

#         offsets.append((idx,idx+len(w)))

#         idx += len(w)

    

#     # START END TOKENS

#     toks = []

#     for i,(a,b) in enumerate(offsets):

#         sm = np.sum(chars[a:b])

#         if sm>0: toks.append(i) 

        

#     s_tok = sentiment_id[train_df.loc[k,'sentiment']]

#     input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

#     attention_mask[k,:len(enc.ids)+5] = 1

#     if len(toks)>0:

#         start_tokens[k,toks[0]+1] = 1

#         end_tokens[k,toks[-1]+1] = 1
ct = test_df.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test_df.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test_df.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test_df.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask_t[k,:len(enc.ids)+5] = 1
def scheduler(epoch):

    return 3e-5 * 0.2**epoch
def build_model():

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)



    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids,attention_mask=att,token_type_ids=tok)

    

    

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



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)



    return model
n_splits = 5
preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

DISPLAY=1

for i in range(10):

    print('#'*25)

    print('### MODEL %i'%(i+1))

    print('#'*25)

    

    K.clear_session()

    model = build_model()

    model.load_weights('../input/model4/v4-roberta-%i.h5'%i)



    print('Predicting Test...')

    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/n_splits

    preds_end += preds[1]/n_splits
all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test_df.loc[k,'text']

    else:

        text1 = " "+" ".join(test_df.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-1:b])

    all.append(st)
test_df['selected_text'] = all

test_df[['textID','selected_text']].to_csv('submission.csv',index=False)