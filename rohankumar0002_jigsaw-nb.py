import pandas as pd

import numpy as np

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Bidirectional, Flatten, Activation, Embedding, Concatenate, Input, Dense, Dropout, MaxPool2D

from keras.layers import Reshape, Flatten, Conv1D, MaxPool1D, Embedding,BatchNormalization, LSTM,merge, Conv2D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Sequential, Model

import keras.utils as ku 

import numpy as np

from keras import regularizers

import pickle as pkl

tokenizer = Tokenizer()

from html import unescape

from nltk.stem import SnowballStemmer

import re

from nltk.corpus import stopwords

import sys

sys.stdout.write('hello')
#stop=set(stopwords.words('english'))

#stem=SnowballStemmer('english')

df=pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')#, sep='\t')

df_un=pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')#, sep='\t')

df['bin'] = df.target>=0.5
def re_clean(text):

    #c.append('')

    #if len(c)%1000==0:

        #pass

        #print(len(c))

    text=text.lower()

    symbol = """!#$%^&*();:\t\\\"!\{\}\[\]<>-\?\-\\\"—\.,1234567890"""

    text=re.sub("\'ll", ' will', text)

    text=re.sub("\'ve", ' have', text)

    text=re.sub("\'s", ' is', text)

    text=re.sub('[{}]'.format(symbol),' ', text)

    #text=re.sub('[\W]',' ', text)

    text=re.sub('\n',' ', text)

    text=re.sub(' +',' ', text)

    text=re.sub('^\s', '', text)

    #text=' '.join([stem.stem(i) for i in text.split() if not i in stop])

    #text=[i for i in text.split() if i not in stop]

    return text #' '.join(text)
sys.stdout.write('Cleaning start')

df['text']=df.comment_text.apply(re_clean)

df_un['text']=df_un.comment_text.apply(re_clean)

sys.stdout.write('Cleaning done')
!cd /kaggle/input/glove840b300dtxt && ls
%%time

tokenizer.fit_on_texts(list(df.text)+list(df_un.text))

sys.stdout.write('tokenizing done!')
m=220
%%time

X_train_seq_trunc = tokenizer.texts_to_sequences(df.text)



X_train_seq_trunc = pad_sequences(X_train_seq_trunc, maxlen=m, truncating='post', padding='post')

#from sklearn.model_selection import train_test_split

y_train = df.bin

#X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train, test_size=0.1, random_state=37)



#assert X_valid_emb.shape[0] == y_valid_emb.shape[0]

#assert X_train_emb.shape[0] == y_train_emb.shape[0]



#print('Shape of validation set:',X_valid_emb.shape)

#del X_train_seq_trunc

del df



sys.stdout.write('sequencing done!')
tokenizer.word_index
import gc

gc.collect()
%%time

NB_WORDS=len(tokenizer.word_index)

GLOVE_DIM=300





emb_matrix = np.zeros((NB_WORDS+1, GLOVE_DIM))

def emb():

    with open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt') as f:

        for n, i in enumerate(f):

            i = i.split()

            if tokenizer.word_index.get(' '.join(i[:-300])):

                #print(' '.join(i[:-300]))

                emb_matrix[tokenizer.word_index.get(' '.join(i[:-300]))] = np.array(i[-300:])

            if n%500000==0:

                sys.stdout.write(n)
emb()
emb_matrix
num_filters = 256

filter_sizes = [3,4,5]

dropout = 0.5

embedding_dim = 300

vocabulary_size = NB_WORDS

sequence_length = m
inp1=Input((m,))

emb=Embedding(NB_WORDS+1,embedding_dim, weights=[emb_matrix],trainable=False)(inp1)

#lstm1=Bidirectional(LSTM(128, return_sequences=True))(emb)

#drop1=Dropout(dropout)(lstm1)

#lstm2=Bidirectional(LSTM(128, return_sequences=True))(drop1)

#drop2=Dropout(dropout)(lstm2)

#lstm2=Bidirectional(LSTM(128, return_sequences=False))(drop2)

#dense_lstm=Dense(128, activation='relu')(lstm2)

reshape = Reshape((sequence_length,embedding_dim,1))(emb)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)

maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)

maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])#, maxpool_3])

flatten = Flatten()(concatenated_tensor)

#dense_con=Dense(128, activation='relu')(flatten)

#conc=conc=merge.multiply(([dense_con, dense_lstm]))

batch1=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', 

                              gamma_initializer='ones', moving_mean_initializer='zeros', 

                              moving_variance_initializer='ones', beta_regularizer=None,

                              gamma_regularizer=None, beta_constraint=None,

                              gamma_constraint=None)(flatten)

dense3=Dense(128, activation='relu')(batch1)

drop4=Dropout(0.5)(dense3)

dense4=Dense(128, activation='relu')(drop4)

"""batch2=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', 

                          gamma_initializer='ones', moving_mean_initializer='zeros', 

                          moving_variance_initializer='ones', beta_regularizer=None,

                          gamma_regularizer=None, beta_constraint=None,

                          gamma_constraint=None)(dense4)"""

#conc=Concatenate()([dense7, out1])

drop5=Dropout(0.5)(dense4)

dense7=Dense(128, activation='relu')(drop5)

out=Dense(1, activation='sigmoid')(dense7)

model= Model(inp1, out)
from keras.optimizers import Adam

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)



checkpoint = ModelCheckpoint('jigsaw.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

model.compile(loss='binary_crossentropy', optimizer=adam,

              metrics=['accuracy'], )



gc.collect()
sys.stdout.write('Training...')

history=model.fit(X_train_seq_trunc

                       , y_train

                       , epochs=8

                       , batch_size=2048

                       , validation_split= 0.1 #(X_valid_emb, y_valid_emb)

                       , verbose=1, callbacks=[checkpoint])

sys.stdout.write("Trained!")
X_test_seq_trunc = tokenizer.texts_to_sequences(df_un.text)

X_test_seq_trunc = pad_sequences(X_test_seq_trunc, maxlen=m, truncating='post', padding='post')

gc.collect()
from keras.models import load_model

model=load_model('jigsaw.hdf5')

pre=model.predict(X_test_seq_trunc, verbose=1)

df3=pd.DataFrame(pre, index=df_un.id, columns=['prediction'])

df3.to_csv('submission.csv')

sys.stdout.write('Done!')