import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline



from keras.preprocessing import sequence,text

from keras.models import Sequential

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM, GRU,SimpleRNN

from keras.layers.core import Dense, Activation, Dropout

import re,string

#!pip install pyspellchecker

#from spellchecker import SpellChecker

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import matplotlib.pyplot as plt



try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)



train_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")



test_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")

test_data.columns = ['id','comment_text','lang']

validation_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
"""

Drop the other columns in the training data 

"""



train_data.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)

len(train_data)
"""

Maximum Number of words in Comments

"""

train_data['comment_text'].apply(lambda x: len(str(x).split())).max()
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



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



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)





"""

train_data['text']=train_data['comment_text'].apply(lambda x : remove_URL(x))

train_data['text']=train_data['comment_text'].apply(lambda x : remove_html(x))

train_data['text']=train_data['comment_text'].apply(lambda x: remove_emoji(x))

train_data['text']=train_data['comment_text'].apply(lambda x : remove_punct(x))



test_data['text']=test_data['content'].apply(lambda x : remove_URL(x))

test_data['text']=test_data['content'].apply(lambda x : remove_html(x))

test_data['text']=test_data['content'].apply(lambda x: remove_emoji(x))

test_data['text']=test_data['content'].apply(lambda x : remove_punct(x))

df=pd.concat([train_data,test_data])

df['text']=df['comment_text'].apply(lambda x : remove_URL(x))

df['text']=df['comment_text'].apply(lambda x : remove_html(x))

df['text']=df['comment_text'].apply(lambda x: remove_emoji(x))

df['text']=df['comment_text'].apply(lambda x : remove_punct(x))







"""



for dataset in [train_data, test_data]:

    

    dataset['text']=dataset['comment_text'].apply(lambda x : remove_URL(x))

    dataset['text']=dataset['comment_text'].apply(lambda x : remove_html(x))

    dataset['text']=dataset['comment_text'].apply(lambda x: remove_emoji(x))

    dataset['text']=dataset['comment_text'].apply(lambda x : remove_punct(x))



    

    







Y = train_data['toxic']
"""

Tokenization

"""

max_length = 1500

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_data['comment_text'])



train_tokenized = tokenizer.texts_to_sequences(train_data['comment_text'])

test_tokenized = tokenizer.texts_to_sequences(test_data['comment_text'])



X = pad_sequences(train_tokenized, maxlen=max_length)

X_ = pad_sequences(test_tokenized, maxlen=max_length)
word_index = tokenizer.word_index

word_index
# Load the GloVe vectors in a dictionary:





 

embeddings_index = {}

f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')

for line in tqdm(f):

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray([float(val) for val in values[1:]])

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))



num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



# create an embedding matrix for the words we have in the dataset

embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

 
model = Sequential()

model.add(Embedding(len(word_index) + 1,

                 300,

                 weights=[embedding_matrix],

                 input_length=1500,

                 trainable=False))



model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()
model.fit(X, Y, batch_size=1024, epochs =2)
pred = model.predict(X_)

temp = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

temp['toxic'] = pred

temp.to_csv('submission.csv', index=False)
"""      

def roc_auc(predictions,target):

    '''

    This methods returns the AUC Score when given the Predictions

    and Labels

    '''

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc

"""

 
"""

Data Preparation



xtrain, xvalid, ytrain, yvalid = train_test_split(train_data.comment_text.values, train_data.toxic.values, 

                                                  stratify=train_data.toxic.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)

xtrain

"""
"""

token = text.Tokenizer(num_words=None)

max_len = 1500

token.fit_on_texts(list(xtrain) + list(xvalid))

print(token.word_index)

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)



xtrain_pad = sequence.pad_sequences(xtrain_seq,max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq,max_len)



word_index = token.word_index

"""

"""

model = Sequential()

model.add(Embedding(len(word_index)+1, 300, input_length=max_len))

model.add(SimpleRNN(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""

"""

model.fit(xtrain_pad, ytrain, nb_epoch=2, batch_size=64) 

scores = model.predict(xvalid_pad)

print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))

"""

"""

token = text.Tokenizer(num_words=None)

max_len = 1500

token.fit_on_texts(list(xtrain) + list(xvalid))

print(token.word_index)

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)



xtrain_pad = sequence.pad_sequences(xtrain_seq,max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq,max_len)



word_index = token.word_index

"""



"""

Glove for Vectorization



def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['comment_text']):

        #print(tweet)

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        #print(words)

        corpus.append(words)

    return corpus



corpus=create_corpus(df)



"""

"""

MAX_LEN = 1500



tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')



word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))



"""


