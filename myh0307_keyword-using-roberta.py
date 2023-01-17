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

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

import re



import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

from keras.layers import Dense, Flatten, Conv1D, Dropout, Input

from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping

train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train_df.head()
test_df.head()
train_df.isnull().sum()
train_df.dropna(inplace=True)

train_df.reset_index(inplace=True)
train_df.isnull().sum()
def word(text):   

    #line = re.findall(r'[a-zA-Z0-9]+', text)

    review_word = BeautifulSoup(text).get_text()

    review_word = re.sub(r'[^a-zA-Z]',' ', review_word)

    words = review_word.lower().split()

    stops = set(stopwords.words('english'))

    words = [w for w in words if w not in stops]

    

    return ' '.join(words)

train_df['clean_text']=train_df['text'].apply(lambda x : word(x))
import emoji

def f_emoji(text):

    sentences=[]

    emoji_text = emoji.demojize(text)

  #  line = re.findall(r'\:(.*?)\:',emoji_text)

    line = re.sub(":","",emoji_text)

    for sentence in line:

  #      sett = re.findall(r'[^\_]+',sentence)

        sett = re.sub("_"," ",sentence)

        sentences.append(' '.join(sett))

  #      sentences.append(' '.join(sentence)) 

    return ''.join(sentences)



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





def remove_email(text):  # text 안에서 email 제거하기

    line = re.compile(r'[\w\.-]+@[\w\.-]+')



    return line.sub(r'',text)



def remove_hash(text):

    line = re.sub(r'\#','',text)

    return ''.join(line)



def remove_phone_num(text):

    line = re.sub(r'\b\d{10}\b','', text)

    line = re.sub(r'(\d{3})\-(\d{3})\-(\d{4})','',line)

    return ''.join(line)



def remove_year(text):

    line=re.sub(r"\b(19[4-9][0-9]|20[0-1][0-9]|2020)\b",'',text)   

    return ''.join(line)



def remove_url(text):

    url = re.sub(r'http[s]?[\:]?//\S+|www\.\S+\.\S+','',text)

    return ''.join(url)



train_df['c_s_text']= train_df['selected_text'].copy()

train_df['c_s_text']= train_df['c_s_text'].apply(lambda x : remove_url(x))

train_df['c_s_text']= train_df['c_s_text'].apply(lambda x : remove_email(x))

train_df['c_s_text']= train_df['c_s_text'].apply(lambda x : remove_emoji(x))

train_df['c_s_text']= train_df['c_s_text'].apply(lambda x : f_emoji(x))

train_df['c_s_text']= train_df['c_s_text'].apply(lambda x : word(x))

sent_r=train_df.groupby('sentiment')['sentiment'].count()
sent_r.plot(kind='bar')
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



sent=['positive','neutral','negative']

fig, ax = plt.subplots(1,3, figsize=(11,11))

for i,s in enumerate(sent):

    

    tot_token = ''

    tot_token +=' '.join(train_df.loc[train_df['sentiment']==s,'c_s_text'])

    wordcloud = WordCloud(width=300, height=300, background_color='white',

                     stopwords = stopwords, min_font_size=6).generate(tot_token)



    ax[i].imshow(wordcloud)  

    ax[i].set_title(s)

    ax[i].axis('off')

 

train_df.loc[train_df['sentiment']=='negative','c_s_text']
import seaborn as sns



def dist_unique_word(df):

    fig,ax = plt.subplots(1,3, figsize=(12,5))

    for i,s in enumerate(sent):

        new = train_df[train_df['sentiment']==s]['c_s_text'].map(lambda x: len(set(x.split())))

        sns.distplot(new.values, ax=ax[i])

        ax[i].set_title(s)

    fig.suptitle('Distribution of number of unique words')

    fig.show()



dist_unique_word(train_df)
neutral_df=train_df[train_df['sentiment']=='neutral']

negative_df = train_df[train_df['sentiment']=='negative']

positive_df = train_df[train_df['sentiment']=='positive']



print('neutral : text vs selected text equivalent rate : {:.2f}'.format(sum(neutral_df['clean_text']==neutral_df['c_s_text'])/len(neutral_df)*100))

print('negative : text vs selected text equivalent rate : {:.2f}'.format(sum(negative_df['clean_text']==negative_df['c_s_text'])/len(negative_df)*100))

print('neutral : text vs selected text equivalent rate : {:.2f}'.format(sum(positive_df['clean_text']==positive_df['c_s_text'])/len(positive_df)*100))
from sklearn.feature_extraction.text import CountVectorizer



def get_top_ngram(corpus, n):

    vec = CountVectorizer(ngram_range=(n,n), analyzer='word', max_features=5000)

    bag_of_words = vec.fit_transform(train_df[train_df['sentiment']==corpus]['c_s_text'])

    sum_words=bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0,idx])for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x:x[1], reverse=True)

    return words_freq[:20]



top_pos=get_top_ngram('positive',3)

top_neu=get_top_ngram('neutral',3)

top_neg=get_top_ngram('negative',3)

print('Top Positive words: ',top_pos)

print('Top Neutral words: ', top_neu)

print('Top Negative words: ', top_neg)
max_len = 150



vocab_path = '/kaggle/input/roberta-base/vocab.json'

merge_path = '/kaggle/input/roberta-base/merges.txt'

tokenizer = tokenizers.ByteLevelBPETokenizer(

            vocab_file = vocab_path,

            merges_file = merge_path,

            lowercase =True,

            add_prefix_space=True

)



sentiment_id = {'positive':tokenizer.encode('positive').ids[0], 'negative':tokenizer.encode('negative').ids[0], 'neutral':tokenizer.encode('neutral').ids[0]}



train_df.reset_index(inplace=True)
# input data formating for training

tot_tw = train_df.shape[0]



input_ids = np.ones((tot_tw, max_len), dtype='int32')

attention_mask = np.zeros((tot_tw, max_len), dtype='int32')

token_type_ids = np.zeros((tot_tw, max_len), dtype='int32')

start_mask = np.zeros((tot_tw, max_len), dtype='int32')

end_mask = np.zeros((tot_tw, max_len), dtype='int32')



for i in range(tot_tw):

    set1 = " "+" ".join(train_df.loc[i,'text'].split())

    set2 = " ".join(train_df.loc[i,'selected_text'].split())

    idx = set1.find(set2)

    set2_loc = np.zeros((len(set1)))

    set2_loc[idx:idx+len(set2)]=1

    if set1[idx-1]==" ":

        set2_loc[idx-1]=1

  

    enc_set1 = tokenizer.encode(set1)



    selected_text_token_idx=[]

    for k,(a,b) in enumerate(enc_set1.offsets):

        sm = np.sum(set2_loc[a:b]) 

        if sm > 0:

            selected_text_token_idx.append(k)



    senti_token = sentiment_id[train_df.loc[i,'sentiment']]

    input_ids[i,:len(enc_set1.ids)+5] = [0]+enc_set1.ids+[2,2]+[senti_token]+[2] 

    attention_mask[i,:len(enc_set1.ids)+5]=1



    if len(selected_text_token_idx) > 0:

        start_mask[i,selected_text_token_idx[0]+1]=1

        end_mask[i, selected_text_token_idx[-1]+1]=1
#input data formating for testing



tot_test_tw = test_df.shape[0]



input_ids_t = np.ones((tot_test_tw,max_len), dtype='int32')

attention_mask_t = np.zeros((tot_test_tw,max_len), dtype='int32')

token_type_ids_t = np.zeros((tot_test_tw,max_len), dtype='int32')



for i in range(tot_test_tw):

    set1 = " "+" ".join(test_df.loc[i,'text'].split())

    enc_set1 = tokenizer.encode(set1)



    s_token = sentiment_id[test_df.loc[i,'sentiment']]

    input_ids_t[i,:len(enc_set1.ids)+5]=[0]+enc_set1.ids+[2,2]+[s_token]+[2]

    attention_mask_t[i,:len(enc_set1.ids)+5]=1
from keras.layers import Dense, Flatten, Conv1D, Dropout, Input

from keras.models import Model

def build_model():

    ids = tf.keras.layers.Input((max_len,), dtype=tf.int32)

    att = tf.keras.layers.Input((max_len,), dtype=tf.int32)

    tok =  tf.keras.layers.Input((max_len,), dtype=tf.int32) 



    config_path = RobertaConfig.from_pretrained('/kaggle/input/prerobertabase/config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained('/kaggle/input/prerobertabase/pretrained-roberta-base.h5', config=config_path)

    x = bert_model(ids, attention_mask = att, token_type_ids=tok)



    

    x1 =  tf.keras.layers.Dropout(0.1)(x[0])

    x1 =  tf.keras.layers.LSTM(1024, return_sequences=True)(x1)

    x1 =  tf.keras.layers.Conv1D(128,2, padding='same')(x1)

    x1 =  tf.keras.layers.LeakyReLU()(x1)

  #x1 =  tf.keras.layers.Conv1D(256,2, padding='same')(x1)

  #x1 =  tf.keras.layers.LeakyReLU()(x1)

  #x1 =  tf.keras.layers.Conv1D(128,2, padding='same')(x1)

    x1 =  tf.keras.layers.Conv1D(16,2, padding='same')(x1)

    x1 =  tf.keras.layers.LeakyReLU()(x1)

  #x1 =  tf.keras.layers.Conv1D(1,1)(x1)

    x1 =  tf.keras.layers.Dense(1)(x1)

    x1 =  tf.keras.layers.Flatten()(x1)

    x1 =  tf.keras.layers.Activation('softmax')(x1)



    x2 =  tf.keras.layers.Dropout(0.1)(x[0])

    x2 =  tf.keras.layers.LSTM(1024, return_sequences=True)(x2)

    x2 =  tf.keras.layers.Conv1D(128,2, padding='same')(x2)

    x2 =  tf.keras.layers.LeakyReLU()(x2)

  #x2 =  tf.keras.layers.Conv1D(256,2, padding='same')(x2)

  #x2 =  tf.keras.layers.LeakyReLU()(x2)

  #x2 =  tf.keras.layers.Conv1D(128,2, padding='same')(x2)

    x2 =  tf.keras.layers.Conv1D(16,2, padding='same')(x2)

    x2 =  tf.keras.layers.LeakyReLU()(x2)

  #x2 =  tf.keras.layers.Conv1D(1,1)(x2)

    x2 =  tf.keras.layers.Dense(1)(x2)

    x2 =  tf.keras.layers.Flatten()(x2)

    x2 =  tf.keras.layers.Activation('softmax')(x2)

    

    

    #x1 =  tf.keras.layers.Dropout(0.1)(x[0])

    #x1 =  tf.keras.layers.Conv1D(16,1, padding='same')(x1)

    #x1 =  tf.keras.layers.Conv1D(16,1, padding='same')(x1)

    #x1 =  tf.keras.layers.Activation('relu')(x1)

    #x1 =  tf.keras.layers.Dropout(0.1)(x1)

    #x1 =  tf.keras.layers.Conv1D(1,1)(x1)

    #x1 =  tf.keras.layers.Activation('relu')(x1)

    #x1 =  tf.keras.layers.Flatten()(x1)

    #x1 =  tf.keras.layers.Activation('softmax')(x1)



    #x2 =  tf.keras.layers.Dropout(0.1)(x[0])

    #x2 =  tf.keras.layers.Conv1D(16,1, padding='same')(x2)

    #x2 =  tf.keras.layers.Conv1D(16,1, padding='same')(x2)

    #x2 =  tf.keras.layers.Activation('relu')(x2)

    #x2 =  tf.keras.layers.Dropout(0.1)(x2)

    #x2 =  tf.keras.layers.Conv1D(1,1)(x2)

    #x2 =  tf.keras.layers.Activation('relu')(x2)

    #x2 =  tf.keras.layers.Flatten()(x2)

    #x2 =  tf.keras.layers.Activation('softmax')(x2)



    model =  tf.keras.models.Model(inputs=[ids,att,tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)



    return model
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
from keras.callbacks import ModelCheckpoint, EarlyStopping

from transformers import TFRobertaModel



jac =[]; DISPLAY=1

#oof_start = np.zeros((input_ids.shape[0],max_len))

#oof_end = np.zeros((input_ids.shape[0],max_len))

preds_start= np.zeros((input_ids_t.shape[0],max_len))

preds_end= np.zeros((input_ids_t.shape[0],max_len))



#skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)

#for fold,(idxT, idxV) in enumerate(skfold.split(input_ids, train_df.sentiment.values)):



#    print('### FOLD %i'%(fold+1))



    

#    K.clear_session()

#    model = build_model()

model = build_model()

#    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

#    mc = tf.keras.callbacks.ModelCheckpoint('sample_data/pre-roberta.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_freq='epoch', save_weights_only=True)

#    model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT]], [start_mask[idxT,],end_mask[idxT,]], epochs=12, batch_size=32, verbose=DISPLAY, callbacks=[es,mc], validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],[start_mask[idxV,],end_mask[idxV,]]))

    

#    print('Loading model ....')

#    model.load_weights('sample_data/pre-roberta.h5')

print('Loading model ....')

model.load_weights('/kaggle/input/preroberta8/pre-roberta-v10.h5')



#    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

#    preds_start += preds[0]/skfold.n_splits

#    preds_end += preds[1]/skfold.n_splits

    

#    all = []

#    for k in idxV:

#        a = np.argmax(oof_start[k,])

#        b = np.argmax(oof_end[k,])

#        if a>b: 

#            st = train_df.loc[k,'text'] # IMPROVE CV/LB with better choice here

#        else:

#            text1 = " "+" ".join(train_df.loc[k,'text'].split())

#            enc = tokenizer.encode(text1)

#            st = tokenizer.decode(enc.ids[a-1:b])

#        all.append(jaccard(st,train_df.loc[k,'selected_text']))

#    jac.append(np.mean(all))

#    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

#    print()



print('-'*15)

print('- RESULT -')

print('-'*15)

preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

preds_start = preds[0]

preds_end = preds[1]

  

all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test_df.loc[k,'text'] # IMPROVE CV/LB with better choice here

    else:

        text1 = " "+" ".join(test_df.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-1:b])

    all.append(st)

test_df['selected_text']=all

test_df[['textID','selected_text']].to_csv('submission.csv', index=False)
model.summary()