# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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
!ls /kaggle/working
train = pd.read_csv('/kaggle/working/train.tsv',sep = '\t')

test = pd.read_csv('/kaggle/working/test.tsv',sep = '\t')
sample_submsission =  pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
train.head()
print('Total number of original reviews are:', len(pd.unique(train['SentenceId'])))
train.shape, test.shape
#let's check some random reviews

indices = np.random.randint(0,train.shape[0],5)

for index in indices:

    print(train['Phrase'][index])

    print(train['Sentiment'][index])
#These reviews are break down in small phrases 
#let's check some review by id

train[train['SentenceId']==10]
train[train['SentenceId']==187]
print('Average phrases per sentence in Train Data are: {:.0f}' .format(train.groupby(['SentenceId'])['Phrase'].count().mean()))

print('Average phrases per sentence in Test Data are: {:.0f}' .format(test.groupby(['SentenceId'])['Phrase'].count().mean()))
print('Total phrases in Train Data is {} and total sentences are {}' .format(train.shape[0],len(pd.unique(train['SentenceId']))))

print('Total phrases in Test Data is {} and total sentences are {}' .format(test.shape[0],len(pd.unique(test['SentenceId']))))
print('Average words in phrases in Train Data is {:f}' .format(train['Phrase'].apply(lambda x: len(x.split())).mean()))

print('Average words in phrases in Test Data is {:f}' .format(test['Phrase'].apply(lambda x: len(x.split())).mean()))
print('Maximum words in phrases in Train Data is {:f}' .format(train['Phrase'].apply(lambda x: len(x.split())).max()))

print('Maximum words in phrases in Test Data is {:f}' .format(test['Phrase'].apply(lambda x: len(x.split())).max()))
print('Minimum words in phrases in Train Data is {:f}' .format(train['Phrase'].apply(lambda x: len(x.split())).min()))

print('Minimum words in phrases in Test Data is {:f}' .format(test['Phrase'].apply(lambda x: len(x.split())).min()))
#removing empty data from train data

to_remove = []

for i,row in train.iterrows():

    if(len(row['Phrase'].split())== 0):

        to_remove.append(i)

print(len(to_remove))

train.drop(to_remove,inplace = True)
#checking again minimun length of phrase

print('Minimum words in phrases in Train Data is {:f}' .format(train['Phrase'].apply(lambda x: len(x.split())).min()))
#let's plot number of reviews

plt.figure(figsize = (12,10))

train['Sentiment'].value_counts().sort_index().plot(kind = 'bar')

plt.xlabel('Review')

plt.ylabel('Count')

plt.show()
tokenizer = Tokenizer()
full_text = list(test['Phrase'].values) + list(train['Phrase'].values)
tokenizer.fit_on_texts(full_text)
#let's print some word counts

i = 0

for k,v in dict(tokenizer.word_counts).items():

    print(k,v)

    if i== 10:

        break

    i+=1
#total phrases

print(tokenizer.document_count)
#word index for one hot encoding

#let's print some

i = 0

for k,v in tokenizer.word_index.items():

    print(k,v)

    if i== 10:

        break

    i+=1
print('Total number of unique words in all of data : {}'.format( max(tokenizer.word_index.values())))
Most_used_words = dict(tokenizer.word_counts)

print('Most used words are:')

sorted(Most_used_words.items() ,key =lambda x:x[1], reverse = True)[:10]
#dividing train data in training and validation data

X_train, X_valid, y_train, y_valid = train_test_split(train['Phrase'],train['Sentiment'],test_size = .1)
print(X_train.shape,y_train.shape)

print(X_valid.shape,y_valid.shape)
X_train = tokenizer.texts_to_sequences(X_train)

X_valid = tokenizer.texts_to_sequences(X_valid)

X_test = tokenizer.texts_to_sequences(test['Phrase'])
#example

print(X_train[0])
#let's use maximum sequence length of 40

max_len = 40

#using default pre padding. if phrase length is more than 40, it is truncated from starting.

X_train = sequence.pad_sequences(X_train, maxlen=max_len)

X_valid = sequence.pad_sequences(X_valid, maxlen=max_len)

X_test = sequence.pad_sequences(X_test, maxlen=max_len)

print(X_train.shape,X_valid.shape,X_test.shape)
y_train = to_categorical(y_train)

y_valid = to_categorical(y_valid)

print(y_train.shape,y_valid.shape)

max_features = 17780 #using all unique words

embedding_dim = 150

num_classes = 5

batch_size = 64
#callbacks

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

reduce_lr =  ReduceLROnPlateau(monitor='val_loss',verbose=1, factor=.1,patience=5)

checkpointer = ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model = Sequential()

model.add(Embedding(max_features + 1, embedding_dim, input_length= max_len, mask_zero = True))        #input dim is max_features + 1 because 0 index is used in padding.

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

plt.plot(history['loss'])

plt.plot(history['val_loss'])



ticks = list(range(len(history['loss'])+1)) # we need integers in x axis (epochs)

plt.xticks(ticks)



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
#loading the best weights

model.load_weights('model.hdf5')
predictions = model.predict(X_test)
y_test = predictions.argmax(axis = 1) 

y_test.shape
sample_submsission.Sentiment = y_test
sample_submsission.to_csv('submission.csv',index=False)