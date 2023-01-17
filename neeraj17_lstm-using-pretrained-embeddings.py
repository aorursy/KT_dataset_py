import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,LSTM,Dropout,Embedding

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

from keras.optimizers import Adam
#to print very long sentences in pandas df

pd.set_option('display.max_colwidth', -1)
!unzip  ../input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip

!unzip  ../input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip
train = pd.read_csv('/kaggle/working/train.tsv',sep = '\t')

test = pd.read_csv('/kaggle/working/test.tsv',sep = '\t')

sample_submsission =  pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
train.head()
to_remove = []

for i,row in train.iterrows():

    if(len(row['Phrase'].split())== 0):

        to_remove.append(i)

print(len(to_remove))

train.drop(to_remove,inplace = True)
tokenizer = Tokenizer()
full_text = list(test['Phrase'].values) + list(train['Phrase'].values)
tokenizer.fit_on_texts(full_text)
X_train, X_valid, y_train, y_valid = train_test_split(train['Phrase'],train['Sentiment'],test_size = .1)
print(X_train.shape,y_train.shape)

print(X_valid.shape,y_valid.shape)
X_train = tokenizer.texts_to_sequences(X_train)

X_valid = tokenizer.texts_to_sequences(X_valid)

X_test = tokenizer.texts_to_sequences(test['Phrase'])
max_len = 40

#using default pre padding. if phrase length is more than 40, it is truncated from starting.

X_train = sequence.pad_sequences(X_train, maxlen=max_len)

X_valid = sequence.pad_sequences(X_valid, maxlen=max_len)

X_test = sequence.pad_sequences(X_test, maxlen=max_len)

print(X_train.shape,X_valid.shape,X_test.shape)
y_train = to_categorical(y_train)

y_valid = to_categorical(y_valid)

print(y_train.shape,y_valid.shape)
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
max_features = 17780 #using all unique words

embedding_dim = 300

num_classes = 5

batch_size = 64
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))



word_index = tokenizer.word_index

nb_words = len(word_index)

embedding_matrix = np.zeros((nb_words + 1, embedding_dim))

for word, i in word_index.items():

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
#callbacks

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

reduce_lr =  ReduceLROnPlateau(monitor='val_loss',verbose=1, factor=.1,patience=5)

checkpointer = ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model = Sequential()

model.add(Embedding(max_features + 1, embedding_dim, input_length= max_len, mask_zero = True, weights = [embedding_matrix], trainable = False)) #using pre-trained embeddings

model.add(LSTM(100,dropout=0.6, recurrent_dropout=0.5,return_sequences=True))                         #returning full sequence for next layer, also using recurrent output

model.add(LSTM(64,dropout=0.6, recurrent_dropout=0.5,return_sequences=False))                         #returning only last output.  

model.add(Dense(num_classes,activation='softmax'))                                                    #final output



model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, validation_data=(X_valid, y_valid),epochs=50, batch_size=batch_size, verbose=1,callbacks = [es,reduce_lr,checkpointer])
#let's plot losses



history = model.history.history

# list all data in history

#print(history.keys())

# summarize history for accuracy

plt.figure(figsize = (12,8))

plt.plot(history['loss'])

plt.plot(history['val_loss'])



ticks = list(range(len(history['loss'])+1)) # we need integers in x axis (epochs)

plt.xticks(ticks)



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
model.load_weights('model.hdf5')
predictions = model.predict(X_test)
y_test = predictions.argmax(axis = 1) 

y_test.shape
sample_submsission.Sentiment = y_test
sample_submsission.to_csv('submission.csv',index=False)