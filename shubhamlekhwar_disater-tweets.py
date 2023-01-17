import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re

import string

import gc

import nltk

from tensorflow.keras import Model

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Embedding, Dense, GRU, Dropout, Activation, Bidirectional, LSTM, Attention, Input, Flatten, RepeatVector, Permute, Multiply, Lambda, dot, concatenate

from sklearn.model_selection import train_test_split

from tensorflow.compat.v1.keras.initializers import Constant

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv('../input/nlp-getting-started/train.csv')

test_df=pd.read_csv('../input/nlp-getting-started/test.csv')
train_df
test_df
count=train_df['target'].value_counts()

sns.barplot(count.index, count.values )
def prepprocessing(text):

    text=text.replace(u'[#:}{[]/\']',' ')

    text=text.replace(u'?', '')

    text=text.replace(u'_', '')

    text=text.translate(str.maketrans('', '', string.punctuation))

    text=text.strip()

    text=text.lower()

    return text
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
def remove_stopwords(text):

    stopword=set(stopwords.words('english'))

    tokens=word_tokenize(text)

    filtered_text = [w for w in tokens if not w in stopword] 

    return filtered_text
target=train_df['target']
del train_df['target']
train_size=train_df.shape[0]
df=pd.concat([train_df,test_df])
df.reset_index(inplace=True)
df.drop(['index', 'id', 'keyword', 'location'], axis=1, inplace=True)
df['text']=df['text'].apply(lambda x : remove_html(x))

df['text']=df['text'].apply(lambda x : remove_URL(x))

df['text']=df['text'].apply(lambda x : prepprocessing(x))

# df['text']=df['text'].apply(lambda x :remove_stopwords(x))
t=Tokenizer()

t.fit_on_texts(df['text'])
# df['text'] = df['text'].apply(lambda x : ' '.join(x))
word_count = lambda sentence: len(word_tokenize(sentence))

longest_sentence = max(df['text'], key=word_count)

MAX_LEN = len(word_tokenize(longest_sentence))
VOCAB_SIZE=len(t.word_index) + 1
seq=t.texts_to_sequences(df['text'])
padded_tweet=pad_sequences(seq, MAX_LEN, padding='post', truncating='post')
padded_tweet.shape
tweet_train=padded_tweet[:train_size]
tweet_test=padded_tweet[train_size:]
embedding_dict={}



with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word = values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
embedding_matrix=np.zeros((VOCAB_SIZE,100))



for word,i in t.word_index.items():

    if i < VOCAB_SIZE:

        emb_vec=embedding_dict.get(word)

        if emb_vec is not None:

            embedding_matrix[i]=emb_vec 
from tensorflow.python.keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

def f1(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def attention_3d_block(hidden_states):

    # @author: felixhao28.

    # hidden_states.shape = (batch_size, time_steps, hidden_size)

    hidden_size = int(hidden_states.shape[2])

    # Inside dense layer

    #              hidden_states            dot               W            =>           score_first_part

    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)

    # W is the trainable weight matrix of attention Luong's multiplicative style score

    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)

    #            score_first_part           dot        last_hidden_state     => attention_weights

    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)

    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)

    score = dot([score_first_part, h_t], [2, 1], name='attention_score')

    attention_weights = Activation('softmax', name='attention_weight')(score)

    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)

    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')

    pre_activation = concatenate([context_vector, h_t], name='attention_output')

    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)

    return attention_vector
# def attention_block(inputs):

#     # inputs.shape = (batch_size, time_steps, input_dim)

#     input_dim = int(inputs.shape[2])

#     attention = Dense(1, activation='tanh')(inputs)                             # input shape = batch * time_steps * 1

#     attention = Flatten()(attention)                                            # input shape = batch * time_steps

#     attention = Activation('softmax')(attention)                                # input shape = batch * time_steps

#     attention = RepeatVector(input_dim)(attention)                              # input shape = batch * input_dim * time_steps

#     attention = Permute([2, 1])(attent`a

#                                 ion)                                      # input shape = batch * time_step * input_dim

#     sent_representation = Multiply()([inputs, attention] )              # input shape = batch * time_step * input_dim

#     sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(input_dim,))(sent_representation)              # input shape = batch * input_dim 

#     return sent_representation
def create_model():

    inp = Input(shape=(MAX_LEN,))

    embedding=Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=MAX_LEN)(inp)

    X=Bidirectional(GRU(100, activation='tanh', dropout=0.5, return_sequences=True))(embedding)

    X=attention_3d_block(X)

#     X=Dropout(0.2)(X)

#     X=Bidirectional(GRU(64, activation='tanh', dropout=0.2))(X)

    X=Dense(1, activation='sigmoid')(X)

    model = Model(inputs=inp, outputs=X)

    model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy', f1])

    return model
# def create_model():

#     model=Sequential()

#     embedding=Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=MAX_LEN)

#     model.add(embedding)

#     model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True)))

#     model.add(Bidirectional(LSTM(64, activation='tanh')))

#     model.add(Dropout(0.2))

#     model.add(Dense(1, activation='sigmoid'))

#     model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy', f1])

#     return model
X_train, X_test, y_train, y_test=train_test_split(tweet_train, target, test_size=0.10)
model=create_model()

model.summary()
BATCH_SIZE=32

EPOCHS=50
history=model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test))
print(history.history.keys())
def plot_history(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

plot_history(history)
results=model.predict(tweet_test, batch_size=BATCH_SIZE, verbose=1)
list1=[]

for i in results: 

    if i>0.5:

        list1.append(1)

    else:

        list1.append(0)

        
sub=pd.DataFrame(test_df['id'])
sub['target']=list1
sub.to_csv('prediction4.csv', index=False, sep=',')