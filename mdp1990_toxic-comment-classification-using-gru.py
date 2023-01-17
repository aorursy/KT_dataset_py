# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import tensorflow as tf



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pyplot as pylab

import seaborn as sns



import re

import keras



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU

from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D, GlobalAveragePooling1D, concatenate

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

#rom keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint 

from keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
train_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")



test_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")

test_data.columns = ['id','comment_text','lang']

validation_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

#target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_data.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)
train_data.describe(include='all')
print(pd.isnull(train_data).sum())
for dataset in [train_data, test_data]:

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('\'ll', ' will'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('\'ve', ' have'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('don\'t', ' do not'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('dont', ' do not'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('aren\'t', ' are not'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('won\'t', ' will not'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('wont', ' will not'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('can\'t', ' cannot'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('cant', ' cannot'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('shan\'t', ' shall not'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('shant', ' shall not'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace('\'m', ' am'))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("doesn't", "does not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("doesnt", "does not"))                                                      

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace( "didn't", "did not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace( "didnt", "did not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("hasn't", "has not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("hasnt", "has not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("haven't", "have not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("havent", "have not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("wouldn't", "would not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace( "didn't", "did not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace( "didnt", "did not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("it's" , "it is"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace( "that's" , "that is"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("weren't" , "were not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace("werent" , "were not"))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace(' u ', ' you '))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: x.replace(' U ', ' you '))

    dataset['comment_text'] = dataset['comment_text'].apply(lambda x: re.sub('[\(\)\"\t_\n.,:=!@#$%^&*-/[\]?|1234567890—]', ' ', x).strip())
"""

plt.figure(figsize=(7,7))

plt.title('Correlation of Features & Targets',y=1.05,size=13)

sns.heatmap(train_data[target_columns].astype(float).corr(),linewidths=0.2,vmax=1.0,square=True,annot=True)

plt.show()

"""

#Y = train_data[target_columns]

Y = train_data['toxic']

Y
max_features = 20000

max_length = 100

embed_size = 300

batch_size = 1024

epochs = 2
"""

Tokenization

"""

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(train_data['comment_text'])



train_tokenized = tokenizer.texts_to_sequences(train_data['comment_text'])

test_tokenized = tokenizer.texts_to_sequences(test_data['comment_text'])



X = pad_sequences(train_tokenized, maxlen=max_length)

X_ = pad_sequences(test_tokenized, maxlen=max_length)
"""

Embedding Matrix

"""

embedding_index = {}

with open("/kaggle/input/glove840b300dtxt/glove.840B.300d.txt", encoding='utf8') as f:

    for line in f:

        values = line.rstrip().rsplit(' ')

        embedding_index[values[0]] = np.asarray(values[1:], dtype='float32')



word_index = tokenizer.word_index

num_words = min(max_features, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embed_size))

for word, i in word_index.items():

    if i >= max_features:

        continue



    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
inp = Input(shape=(max_length,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1,recurrent_dropout=0.1))(x)

x = Conv1D(64, kernel_size = 3, padding = "valid", activation="relu")(x)



x = concatenate([GlobalAveragePooling1D()(x), GlobalMaxPool1D()(x)])



x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

model.summary()
Y
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-8)

model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.1,

              callbacks=[reduce_lr])

sumbission_file = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

#sumbission_file = sumbission_file.drop('toxic',axis=1)
sub = model.predict(X_)

#cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



"""

for i in cols:

    sumbission_file[i]=""

"""





sumbission_file['toxic'] = sub

sumbission_file.to_csv('submission.csv', index=False)