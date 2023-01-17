import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os 



%matplotlib inline
print(os.listdir("../working"))
train_data = pd.read_csv("../input/quora-data/quora_train_test/quora_train.csv")
test_data = pd.read_csv("../input/quora-data/quora_train_test/quora_test.csv")
tt = pd.read_csv("../input/quora-data/quora_train_test/quora_test.csv")

print(tt.shape)
tt["question1"].isnull().sum()
tt["question2"].isnull().sum()
train_data.head()
train_data.shape
test_data.shape
test_data.fillna(value = " ",inplace = True)

train_data.fillna(value = " ",inplace = True)
test_data[['question1','question2']].isnull().sum()
import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
def striphtml(data): 

    cleanr = re.compile('<.*?>') 

    cleantext = re.sub(cleanr, ' ', str(data)) 

    return cleantext   
def stripunc(data): 

    return re.sub('[^A-Za-z]+', ' ', str(data), flags=re.MULTILINE|re.DOTALL) 
x = "Hello, World! <how are you?>"
stripunc(x)
striphtml(x)
import nltk

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import word_tokenize



stop_words = set(stopwords.words('english')) 

stemmer = SnowballStemmer("english") 



from tqdm import tqdm
def compute(sent): 

    

    sent = decontracted(sent) 

    sent = striphtml(sent) 

    sent = stripunc(sent) 

    

    words=word_tokenize(str(sent.lower())) 

    

    #Removing all single letter and and stopwords from question 

    sent1=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1)) 

    sent2=' '.join(str(j) for j in words if j not in stop_words and (len(j)!=1)) 

    return sent1, sent2   
clean_stemmed_q1 = []

clean_stemmed_q2 = []

clean_q1 = []

clean_q2 = []

combined_stemmed_text = []

for _, row in tqdm(train_data.iterrows()):

    csq1, cq1 = compute(row['question1'])

    csq2, cq2 = compute(row['question2'])

    clean_stemmed_q1.append(csq1)

    clean_q1.append(cq1)

    clean_stemmed_q2.append(csq2)

    clean_q2.append(cq2)

    combined_stemmed_text.append(csq1+" "+csq2)
clean_stemmed_q1_t = []

clean_stemmed_q2_t = []

clean_q1_t = []

clean_q2_t = []

combined_stemmed_text_t = []

for _, row in tqdm(test_data.iterrows()):

    

    csq1_t, cq1_t = compute(row['question1'])

    csq2_t, cq2_t = compute(row['question2'])

    clean_stemmed_q1_t.append(csq1_t)

    clean_q1_t.append(cq1_t)

    clean_stemmed_q2_t.append(csq2_t)

    clean_q2_t.append(cq2_t)

    combined_stemmed_text_t.append(csq1_t+" "+csq2_t)
test_data["question1"][test_data["question2"].isnull()]
test_data["question2"][test_data["question1"].isnull()]
print(len(clean_stemmed_q1))

print(len(clean_stemmed_q2))

print(len(clean_q1))

print(len(clean_q2))

print(len(combined_stemmed_text))
train_data.head()
train_data['clean_stemmed_q1'] = clean_stemmed_q1

train_data['clean_stemmed_q2'] = clean_stemmed_q2

train_data['clean_q1'] = clean_q1

train_data['clean_q2'] = clean_q2

train_data['combined_stemmed_text'] = combined_stemmed_text
test_data['clean_stemmed_q1_t'] = clean_stemmed_q1_t

test_data['clean_stemmed_q2_t'] = clean_stemmed_q2_t

test_data['clean_q1_t'] = clean_q1_t

test_data['clean_q2_t'] = clean_q2_t

test_data['combined_stemmed_text_t'] = combined_stemmed_text_t
train_data.tail()
test_data.head()
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_data[['clean_q1', 'clean_q2']], train_data['is_duplicate'], test_size=0.1, random_state=42)
test_data = test_data.rename(columns = {'clean_q1_t':'clean_q1','clean_q2_t':'clean_q2'})



X_test = test_data[['clean_q1', 'clean_q2']]

y_test = test_data['is_duplicate']
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)

print(X_test.shape)

print(y_test.shape)
X_train.head()
X_train['text'] = X_train[['clean_q1','clean_q2']].apply(lambda x:str(x[0])+" "+str(x[1]), axis=1)
import tensorflow as tf
import keras

import keras.backend as K
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM

from keras.models import Model
t = Tokenizer()

t.fit_on_texts(X_train['text'].values)
X_train['clean_q1'] = X_train['clean_q1'].astype(str)

X_train['clean_q2'] = X_train['clean_q2'].astype(str)
X_val['clean_q1'] = X_val['clean_q1'].astype(str)

X_val['clean_q2'] = X_val['clean_q2'].astype(str)



X_test['clean_q1'] = X_test['clean_q1'].astype(str)

X_test['clean_q2'] = X_test['clean_q2'].astype(str)
train_q1_seq = t.texts_to_sequences(X_train['clean_q1'].values)
train_q2_seq = t.texts_to_sequences(X_train['clean_q2'].values)

val_q1_seq = t.texts_to_sequences(X_val['clean_q1'].values)

val_q2_seq = t.texts_to_sequences(X_val['clean_q2'].values)

test_q1_seq = t.texts_to_sequences(X_test['clean_q1'].values)

test_q2_seq = t.texts_to_sequences(X_test['clean_q2'].values)
len_vec = [len(sent_vec) for sent_vec in train_q1_seq]
np.max(len_vec)
sns.distplot(len_vec)
len_vec = [len(sent_vec) for sent_vec in train_q2_seq]
np.max(len_vec)
sns.distplot(len_vec)
max_len = 30
train_q1_seq = pad_sequences(train_q1_seq, maxlen=max_len, padding='post')
train_q2_seq = pad_sequences(train_q2_seq, maxlen=max_len, padding='post')

val_q1_seq = pad_sequences(val_q1_seq, maxlen=max_len, padding='post')

val_q2_seq = pad_sequences(val_q2_seq, maxlen=max_len, padding='post')

test_q1_seq = pad_sequences(test_q1_seq, maxlen=max_len, padding='post')

test_q2_seq = pad_sequences(test_q2_seq, maxlen=max_len, padding='post')
print(len(train_q1_seq[0]))

print(len(train_q2_seq[0]))

print(len(val_q1_seq[0]))

print(len(val_q2_seq[0]))

print(len(test_q1_seq[0]))

print(len(test_q1_seq[0]))
import joblib
#https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

embeddings_index = {}

f = open('../input/glove6b50dtxt/glove.6B.50d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
len(embeddings_index.keys())
len(embeddings_index['apple'])
not_present_list = []

vocab_size = len(t.word_index) + 1

print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, len(embeddings_index['no'])))

for word, i in t.word_index.items():

    if word in embeddings_index.keys():

        embedding_vector = embeddings_index.get(word)

    else:

        not_present_list.append(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

    else:

        embedding_matrix[i] = np.zeros(300)

embedding_matrix.shape
len(t.word_index) 
#not_present_list
from keras.regularizers import l2

from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.models import Model



from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense

from keras.initializers import glorot_uniform

from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D
from keras import backend as K



def cosine_distance(vests):

    x, y = vests

    x = K.l2_normalize(x, axis=-1)

    y = K.l2_normalize(y, axis=-1)

    return -K.mean(x * y, axis=-1, keepdims=True)



def cos_dist_output_shape(shapes):

    shape1, shape2 = shapes

    return (shape1[0],1)
from sklearn.metrics import roc_auc_score



def auroc(y_true, y_pred):

    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

input_1 = Input(shape=(train_q1_seq.shape[1],))

input_2 = Input(shape=(train_q2_seq.shape[1],))





common_embed = Embedding(name="synopsis_embedd",input_dim =len(t.word_index)+1, 

                       output_dim=len(embeddings_index['no']),weights=[embedding_matrix], 

                       input_length=train_q1_seq.shape[1],trainable=True) 

lstm_1 = common_embed(input_1)

lstm_2 = common_embed(input_2)





common_lstm = LSTM(64,return_sequences=True, activation="relu")

common_lstm_2 = LSTM(64,return_sequences=True, activation="relu")

vector_1 = common_lstm(lstm_1)

vector_1 = common_lstm_2(vector_1)

vector_1 = Flatten()(vector_1)



vector_2 = common_lstm(lstm_2)

vector_2 = common_lstm_2(vector_2)

vector_2 = Flatten()(vector_2)



x3 = Subtract()([vector_1, vector_2])

x3 = Multiply()([x3, x3])



x1_ = Multiply()([vector_1, vector_1])

x2_ = Multiply()([vector_2, vector_2])

x4 = Subtract()([x1_, x2_])

    

    #https://stackoverflow.com/a/51003359/10650182

x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([vector_1, vector_2])

    

conc = Concatenate(axis=-1)([x5,x4, x3])



x = Dense(100, activation="relu", name='conc_layer')(conc)

x = Dropout(0.01)(x)

x = Dense(100, activation="relu", name='dense_layer')(x)

x = Dropout(0.01)(x)

out = Dense(1, activation="sigmoid", name = 'out')(x)



model = Model([input_1, input_2], out)



model.compile(loss="binary_crossentropy", metrics=['acc','AUC'], optimizer=Adam(0.0001))
model.summary()
my_callbacks = [

    tf.keras.callbacks.EarlyStopping(patience=2),

    tf.keras.callbacks.ModelCheckpoint(filepath='../working/model.{epoch:02d}-{val_loss:.2f}.h5'),

    tf.keras.callbacks.TensorBoard(log_dir='./logs'),

]
model.fit([train_q1_seq,train_q2_seq],y_train.values.reshape(-1,1), epochs = 10,

          batch_size=64,validation_data=([val_q1_seq, val_q2_seq],y_val.values.reshape(-1,1)), callbacks=my_callbacks)