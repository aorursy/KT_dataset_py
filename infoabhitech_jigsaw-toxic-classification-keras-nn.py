# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))

print(os.listdir("../input/glove840"))

print(os.listdir("../input/wikinews"))

# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")



from numpy import array

from numpy import asarray

from numpy import zeros



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding

from keras.layers import Input

from keras.layers import Conv1D

from keras.layers import MaxPooling1D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Dense

from keras.optimizers import RMSprop

from keras.models import Model

from keras.models import load_model
df_train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

df_test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

df_train.head()
df_train.dtypes
df_test.head()
print("Number of samples in training data", len(df_train))
# Create dataframe filtering few particular groups only

identity = ['male','female','homosexual_gay_or_lesbian','christian','jewish','muslim','black','white','psychiatric_or_mental_illness']



def funnewdf():

    newdf= pd.DataFrame()

    identity = ['male','female','homosexual_gay_or_lesbian','christian','jewish','muslim','black','white','psychiatric_or_mental_illness']

    for col in identity:

        newdf = pd.concat([newdf, df_train[pd.notnull(df_train[col])]], ignore_index=True)

    return newdf
df_train = funnewdf()

df_train=df_train[['id','target','comment_text','male','female','homosexual_gay_or_lesbian','christian','jewish','muslim','black','white','psychiatric_or_mental_illness']]

df_train = df_train.drop_duplicates().reset_index()
print("Number of samples in training data after filtering -", len(df_train))
df_train['comment_text'] = df_train['comment_text'].astype(str) 

df_test['comment_text'] = df_test['comment_text'].astype(str) 
# Function to convert toxic value of a column to boolean , 1 being most toxic

def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, 1, 0)
# Function to convert into boolean value for whole dataframe

def convert_dataframe_to_bool(df):

    bool_df = df.copy()

    convert_to_bool(bool_df, 'target')

    return bool_df
df_train = convert_dataframe_to_bool(df_train)
df_train.head()
#Initiating Keras tokenizer

tokenizer = Tokenizer()
#Fitting tokenizer on train data and converting tokens to sequence

tokenizer.fit_on_texts(df_train['comment_text'])

ts_train=tokenizer.texts_to_sequences(df_train['comment_text'])

X_train_vectorized=pad_sequences(ts_train,maxlen=800,padding='post')
vocab_size = len(tokenizer.word_index) + 1

print("Size of corpus in terms of number of tokens to be trained" ,len(tokenizer.word_index))
# Using pretrained Glove embedding vector 

embeddings_index = {}

f = open('../input/glove840/glove.840B.300d.txt', encoding='utf8')

for line in f:

    values = line.split()

    word = ''.join(values[:-300])

    coefs = np.asarray(values[-300:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = zeros((vocab_size, 300))

for word, i in tokenizer.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
embedding_matrix.shape
# Applying tokenizer on test data

tokenizer.fit_on_texts(df_test['comment_text'])

ts_test=tokenizer.texts_to_sequences(df_test['comment_text'])

X_test_vectorized=pad_sequences(ts_test,maxlen=800,padding='post')
y_train = df_train['target']
del df_train , df_test , ts_train , ts_test , embeddings_index

import gc

gc.collect()
model = Sequential()

e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=800, trainable=False)

model.add(e)

model.add(Conv1D(256, 2, activation='relu', padding='same'))

model.add(MaxPooling1D(5, padding='same'))

model.add(Conv1D(256, 3, activation='relu', padding='same'))

model.add(MaxPooling1D(5, padding='same'))

model.add(Conv1D(256, 4, activation='relu', padding='same'))

model.add(MaxPooling1D(40, padding='same'))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))

model.add(Dense(2, activation='softmax'))

#model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(lr=0.0005),metrics=['sparse_categorical_accuracy'])

#model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.0005),metrics=['acc'])

print(model.summary())
model.fit(X_train_vectorized, y_train, epochs=3, batch_size=1024, validation_split=0.2, verbose=1)
# Prediction on test data for competition submission

predictions = model.predict(X_test_vectorized)[:,1]
preds = pd.Series(predictions,name="prediction")
df_id = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv",usecols=['id'])
df_submission = pd.concat([df_id, preds], axis=1)
df_submission.head()
df_submission.to_csv("submission.csv", columns = df_submission.columns, index=False)