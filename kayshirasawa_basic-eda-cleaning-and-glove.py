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

stop=set(stopwords.words('english'))

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





import os

#os.listdir('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')

tweet.head(3)
from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

import numpy as np
# tweet = pd.merge(tweet, tweet_supplemental_data)

# test = pd.merge(test, test_supplemental_data)
print('There are {} rows and {} columns in train'.format(tweet.shape[0],tweet.shape[1]))

print('There are {} rows and {} columns in train'.format(test.shape[0],test.shape[1]))
df=pd.concat([tweet,test])

df.shape
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



remove_URL(example)
df['text']=df['text'].apply(lambda x : remove_URL(x))
example = """<div>

<h1>Real or Fake</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

print(remove_html(example))
df['text']=df['text'].apply(lambda x : remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

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



remove_emoji("Omg another Earthquake 😔😔")
df['text']=df['text'].apply(lambda x: remove_emoji(x))

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
!pip install pyspellchecker
from spellchecker import SpellChecker



spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)

        

text = "corect me plese"

correct_spellings(text)
#df['text']=df['text'].apply(lambda x : correct_spellings(x)#)


def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus

        

        
corpus=create_corpus(df)
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=50

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec

            
model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])



model.summary()
print(tweet.shape[0])

train=tweet_pad[:tweet.shape[0]]

test=tweet_pad[tweet.shape[0]:]

print(train[0])

print(test)

train2 = train.copy()
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)
y_pre=model.predict(test)

print(y_pre)

# y_pre=np.round(y_pre).astype(int).reshape(3263)

print(test.shape)
x_pre=model.predict(train2)

print(x_pre)
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
def split(df):

    if 'target' in df:

        y = df['target']

        x = df.drop(columns='target')

    else:

        y = None

        x = df

    return x,y
from datetime import date

tweet_supplemental_data = pd.read_csv('../input/nlpwithdisastertweets-searching-real-tweets/train_tweet.csv')

test_supplemental_data = pd.read_csv('../input/nlpwithdisastertweets-searching-real-tweets/test_tweet.csv')



def preprocess(df):

    df['has_permalink'] = df['permalink'].map(lambda x: 1 if type(x) == str else 0)

    df['has_mentions'] = df['mentions'].map(lambda x: 1 if type(x) == str else 0)

    df['has_hashtags'] = df['hashtags'].map(lambda x: 1 if type(x) == str else 0)

    df['has_geo'] = df['geo'].map(lambda x: 1 if type(x) == str else 0)

    df['has_urls'] = df['urls'].map(lambda x: 1 if type(x) == str else 0)

    df = df.fillna(0)

#     df['date_from_20200901'] = df['date'].map(lambda x: abs(date(*map(int,x[:10].split('-')))-date(2020,9,1)).days if type(x) == str else 0)

    # min_date = min()

    df = df.drop(columns=['permalink', 'username', 'text', 'mentions', 'hashtags', 'geo', 'urls', 'date'])

    return df

tweet_supplemental_data = preprocess(tweet_supplemental_data)

print(tweet_supplemental_data)

tweet_supplemental_data['target'] = tweet['target']

test_supplemental_data = preprocess(test_supplemental_data)



tweet_supplemental_data
clf_lst = [RandomForestClassifier(),LogisticRegression(max_iter=10000),LinearSVC(max_iter=10000)]

clf_num = len(clf_lst)



sub = pd.DataFrame(pd.read_csv('../input/nlp-getting-started/test.csv')['id'])

repeat = 10



print(((tweet_supplemental_data['has_permalink'] == 0) & (tweet_supplemental_data['target'] == 0)).sum())

print(((tweet_supplemental_data['has_permalink'] == 0) & (tweet_supplemental_data['target'] == 1)).sum())

print(((tweet_supplemental_data['has_permalink'] == 1) & (tweet_supplemental_data['target'] == 0)).sum())

print(((tweet_supplemental_data['has_permalink'] == 1) & (tweet_supplemental_data['target'] == 1)).sum())

# tweet_supplemental_data = tweet_supplemental_data.dropna()

# test_supplemental_data = test_supplemental_data.dropna()

total_val = np.zeros(test_supplemental_data['id'].count())



final_tweet = tweet_supplemental_data

final_tweet['keras_result'] = x_pre

final_test = test_supplemental_data

final_test['keras_result'] = y_pre

for i in range(clf_num):

    clf = clf_lst[i]

    scores = []

    %time

    for j in range(repeat):

        train, val = train_test_split(final_tweet, train_size=0.8)

        train_x, train_y = split(train)

        val_x, val_y = split(val)

        

        _ = clf.fit(train_x, train_y)

        val_pred = clf.predict(val_x)

        scores.append(accuracy_score(val_y.values, val_pred))



        test_x, _ = split(final_test)

        test_pred = clf.predict(test_x)

#         print(test_pred.shape)

        total_val += np.array(test_pred)

    sum(scores)/repeat



# print(len(total_val))

total_val = list(map(lambda x: 1 if 0.5 <= x/(clf_num*repeat) else 0, total_val))

sub['target'] = total_val

sub.to_csv('./submission.csv',index=False)
sub.head()