import numpy as np 
import pandas as pd 
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
data_tr1 = pd.read_csv('../input/nnfl-lab-4/train.csv')
data_tr = data_tr1.copy()

data_te1 = pd.read_csv('../input/nnfl-lab-4/test.csv')
data_te = data_te1.copy()
#Find max-length of a sentence
max_len = 0
for i in range(len(data_tr['Sentence1'])):
  if len(data_tr['Sentence1'][i]) > max_len:
    max_len = len(data_tr['Sentence1'][i])
  if len(data_tr['Sentence2'][i]) > max_len:
    max_len = len(data_tr['Sentence2'][i])
print(max_len)
#Instantiate tokenizer object
num_words = 5000
tokenizer_tr = Tokenizer(num_words=num_words, filters='!"#$%&*()+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890',lower=True,split=' ')

#Fit tokenizer on Sentence1, Sentence2 columns of train data
tokenizer_tr.fit_on_texts(data_tr['Sentence1'].values)
tokenizer_tr.fit_on_texts(data_tr['Sentence2'].values)

#Text to Seq: Train data
X_tr_s1 = tokenizer_tr.texts_to_sequences(data_tr['Sentence1'].values)
X_tr_s2 = tokenizer_tr.texts_to_sequences(data_tr['Sentence2'].values)

#Text to Seq: Test data
X_te_s1 = tokenizer_tr.texts_to_sequences(data_te['Sentence1'])
X_te_s2 = tokenizer_tr.texts_to_sequences(data_te['Sentence2'])

#Padding
max_length_of_text = 334
X_tr_s1 = pad_sequences(X_tr_s1, maxlen=max_length_of_text)
X_tr_s2 = pad_sequences(X_tr_s2, maxlen=max_length_of_text)
X_te_s1 = pad_sequences(X_te_s1, maxlen=max_length_of_text)
X_te_s2 = pad_sequences(X_te_s2, maxlen=max_length_of_text)

#Train data info
word_index_tr = tokenizer_tr.word_index
print('Found %s unique tokens in train data.' % len(word_index_tr))
!wget https://nlp.stanford.edu/data/glove.6B.zip
! unzip glove.6B.zip
#Pick pre-trained model of appropriate dimensions
embeddings_index = {}
f = open('glove.6B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors in pretrained word vector model.' % len(embeddings_index))
print('Dimensions of the vector space : ', len(embeddings_index['the']))

#Create embedding matrix
EMBEDDING_DIM = 200
embedding_matrix = np.zeros((len(word_index_tr) + 1, EMBEDDING_DIM))
for word, i in word_index_tr.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
from keras.layers import Conv1D,MaxPooling1D,Flatten,GlobalAveragePooling1D,BatchNormalization,Dropout
from keras.layers import concatenate,Lambda
from keras import backend as K
from keras.regularizers import l2

#Define model
def model_conv1D_(emb_matrix):
  #Define embedding layer
  emb_layer = Embedding(
      input_dim=emb_matrix.shape[0],
      output_dim=emb_matrix.shape[1],
      weights=[emb_matrix],
      input_length=max_length_of_text,
      trainable=True
    )
    
  #Define model input  
  seq1 = Input(shape=(334,))
  seq2 = Input(shape=(334,))

  #Create embedding layer objects  
  emb1 = emb_layer(seq1)
  emb2 = emb_layer(seq2)

  # 1D convolutions that can iterate over the word vectors
  conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
  conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
  conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
  conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
  conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
  conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')
  conv1a = conv1(emb1)
  glob1a = GlobalAveragePooling1D()(conv1a)
  
  #Convolutional layers for Sentence1, Sentence2
  conv1b = conv1(emb2)
  glob1b = GlobalAveragePooling1D()(conv1b)
  conv2a = conv2(emb1)
  glob2a = GlobalAveragePooling1D()(conv2a)
  conv2b = conv2(emb2)
  glob2b = GlobalAveragePooling1D()(conv2b)
  conv3a = conv3(emb1)
  glob3a = GlobalAveragePooling1D()(conv3a)
  conv3b = conv3(emb2)
  glob3b = GlobalAveragePooling1D()(conv3b)
  conv4a = conv4(emb1)
  glob4a = GlobalAveragePooling1D()(conv4a)
  conv4b = conv4(emb2)
  glob4b = GlobalAveragePooling1D()(conv4b)
  conv5a = conv5(emb1)
  glob5a = GlobalAveragePooling1D()(conv5a) 
  conv5b = conv5(emb2)
  glob5b = GlobalAveragePooling1D()(conv5b)
  
  #Merge convolution layers
  mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a])
  mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b])

  #Extract info from merged convolution layers
  diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(544,))([mergea, mergeb])
  mul = Lambda(lambda x: x[0] * x[1], output_shape=(544,))([mergea, mergeb])
  
  #Merge Sentence1, Sentence2 info
  merge = concatenate([diff, mul])

  #Model  
  x = Dropout(0.5)(merge)
  x = BatchNormalization()(x)    
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(32, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = BatchNormalization()(x)
  pred = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=[seq1, seq2], outputs=pred)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
  return model

#Instantiate model object
model = model_conv1D_(embedding_matrix)
#Define target
y = data_tr['Class'].copy()

#Train model
model.fit([X_tr_s1,X_tr_s2],y,batch_size = 20, epochs = 12)
#Obtain predictions for test data
pred = model.predict([X_te_s1,X_te_s2])

#Assign class by thresholding
pred_list = []
for i in pred:
  if i < 0.6:
    pred_list.append(0)
  if i >= 0.6:
    pred_list.append(1)
#Output dataframe
outdf = pd.DataFrame(columns=['ID','Class'])
outdf['ID'] = data_te1['ID'].copy()
outdf['Class'] = pred_list
outdf.to_csv('sub15_clas.csv',index=False)

#Obtain model weights
model.save_weights("sub15_clas.h5")

#Downloadable dataframe
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(outdf)