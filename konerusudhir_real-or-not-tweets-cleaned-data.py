# Tensorflow

import tensorflow as tf

from tensorflow.keras.utils import plot_model

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import average_precision_score, auc, classification_report, confusion_matrix, roc_curve, precision_recall_curve



from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from keras.optimizers import Adam



from nltk.corpus import stopwords

from nltk.util import ngrams



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import nltk

from collections import defaultdict

from collections import  Counter

import re

import gensim

import string

from tqdm import tqdm



tweet= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')

tweet.head(3)
print('There are {} rows and {} columns in train'.format(tweet.shape[0],tweet.shape[1]))

print('There are {} rows and {} columns in train'.format(test.shape[0],test.shape[1]))
x=tweet.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=tweet[tweet['target']==1]['text'].str.len()

ax1.hist(tweet_len,color='red')

ax1.set_title('disaster tweets')

tweet_len=tweet[tweet['target']==0]['text'].str.len()

ax2.hist(tweet_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=tweet[tweet['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='red')

ax1.set_title('disaster tweets')

tweet_len=tweet[tweet['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Words in a tweet')

plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word=tweet[tweet['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')

ax1.set_title('disaster')

word=tweet[tweet['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')

ax2.set_title('Not disaster')

fig.suptitle('Average word length in each tweet')
def create_corpus(target):

    corpus=[]

    

    for x in tweet[tweet['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
corpus=create_corpus(0)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

x,y=zip(*top)

plt.bar(x,y)
corpus=create_corpus(1)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1



top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    





x,y=zip(*top)

plt.bar(x,y)
plt.figure(figsize=(10,5))

corpus=create_corpus(1)



dic=defaultdict(int)

import string

special = string.punctuation

for i in (corpus):

    if i in special:

        dic[i]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y)
plt.figure(figsize=(10,5))

corpus=create_corpus(0)



dic=defaultdict(int)

import string

special = string.punctuation

for i in (corpus):

    if i in special:

        dic[i]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y,color='green')


counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in stop) :

        x.append(word)

        y.append(count)
sns.barplot(x=y,y=x)
def get_top_tweet_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_tweet_bigrams=get_top_tweet_bigrams(tweet['text'])[:10]

x,y=map(list,zip(*top_tweet_bigrams))

sns.barplot(x=y,y=x)
df=pd.concat([tweet,test])

df.shape
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



df['text'] = df['text'].apply(lambda x: remove_URL(x).strip())

df.head(10)
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



df['text'] = df['text'].apply(lambda x: remove_html(x).strip())

df.head(10)
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



# remove_emoji("Omg another Earthquake 😔😔")

df['text'] = df['text'].apply(lambda x: remove_emoji(x).strip())

df.head(10)
def remove_punct(text):

    text  = "".join([char for char in text if char not in string.punctuation])

    text = re.sub('[0-9]+', '', text)

    return text



df['text_punct'] = df['text'].apply(lambda x: remove_punct(x).strip())

df.head(10)
def tokenization(text):

    text = re.split('\W+', text)

    return text



df['text_tokenized'] = df['text_punct'].apply(lambda x: tokenization(x.lower()))

df.head()
stopword = stopwords.words('english')

#stopword.extend(['yr', 'year', 'woman', 'man', 'girl','boy','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',

#               'treatment', 'associated', 'patients', 'may','day', 'case','old'])

def remove_stopwords(text):

    text = [word for word in text if word not in stopword]

    return text

    

df['text_nonstop'] = df['text_tokenized'].apply(lambda x: remove_stopwords(x))

df.head(10)
ps = nltk.PorterStemmer()



def stemming(text):

    text = [ps.stem(word) for word in text]

    return text



df['text_stemmed'] = df['text_nonstop'].apply(lambda x: stemming(x))

df.head()
wn = nltk.WordNetLemmatizer()



def lemmatizer(text):

    text = [wn.lemmatize(word) for word in text]

    return text



df['text_lemmatized'] = df['text_nonstop'].apply(lambda x: lemmatizer(x))

df.head()
# !pip install pyspellchecker
# from spellchecker import SpellChecker



# spell = SpellChecker()

# def correct_spellings(text):

#     corrected_text = []

#     misspelled_words = spell.unknown(text.split())

#     for word in text.split():

#         if word in misspelled_words:

#             corrected_text.append(spell.correction(word))

#         else:

#             corrected_text.append(word)

#     return " ".join(corrected_text)

        

# text = "corect me plese"

# correct_spellings(text)
#df['text']=df['text'].apply(lambda x : correct_spellings(x)#)
df['text_final'] = df['text_lemmatized'].apply(lambda x: " ".join(x))

df.head()
glove_25d_path = "/kaggle/input/glove-embeddings/glove.twitter.27B.25d.txt"

glove_50d_path = "/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.50d.txt"



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



EMBEDDING_DIMENSIONS=25        

embedding_dict={}

with open(glove_25d_path,'r') as f:

    for line in f:

        values=line.split()

        word = values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=33

VOCAB_SIZE = 25000

OOV_TOKEN = "<XXX>"

tokenizer_obj=Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)

tokenizer_obj.fit_on_texts(df['text_final'])

sequences=tokenizer_obj.texts_to_sequences(df['text_final'])



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))


embedding_matrix=np.zeros((VOCAB_SIZE,EMBEDDING_DIMENSIONS))



for word,i in tqdm(word_index.items()):

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec

            
tweet_pad[0][0:]
def get_base_model():

    model=Sequential()



    embedding=Embedding(VOCAB_SIZE,EMBEDDING_DIMENSIONS,embeddings_initializer=Constant(embedding_matrix),

                       input_length=MAX_LEN,trainable=False)



    model.add(embedding)

    model.add(SpatialDropout1D(0.2))

    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))

    optimzer=Adam(learning_rate=2e-3)

    model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

    return model



def get_model_v2():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSIONS, input_length=MAX_LEN, weights=[embedding_matrix], trainable=False))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))) # dropout=0.2, recurrent_dropout=0.2

#     model.add(tf.keras.layers.BatchNormalization())

#     model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(8, activation='relu'))

#     model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



    metrics = ['accuracy', tf.keras.metrics.Recall(),  tf.keras.metrics.Precision(), tf.keras.metrics.AUC()]

    optimzer=tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=metrics)

    

    return model



model = get_model_v2()
model.summary()
train=tweet_pad[:tweet.shape[0]]

test=tweet_pad[tweet.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)



EPOCHS = 10

BATCH_SIZE = 16

history=model.fit(

    X_train,

    y_train, 

    epochs=EPOCHS, 

    batch_size=BATCH_SIZE, 

    validation_data=(X_test,y_test),

    verbose=2

)
model_loss = pd.DataFrame(model.history.history)

# model_loss.head()

model_loss[['loss','val_loss']].plot(ylim=[0,1])

pd.DataFrame(model.history.history).filter(regex="precision|recall", axis=1).plot(ylim=[0,1])
predictions = model.predict_classes(X_test) 

print(classification_report(y_test, predictions, target_names=["Real", "Not Real"]))
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
y_pre=model.predict(test)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})

sub.to_csv('submission.csv', index=False, header=True)