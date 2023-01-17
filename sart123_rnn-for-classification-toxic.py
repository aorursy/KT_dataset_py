# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers.core import Dense, Flatten, Activation

from keras.layers.wrappers import TimeDistributed

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test_df=pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

train_df.describe()
import seaborn as sns

sns.distplot(train_df['toxic'],kde=False)

'''

sns.distplot(train_df['severe_toxic'])

sns.distplot(train_df['obscene'])

sns.distplot(train_df['threat'])

sns.distplot(train_df['insult'])

sns.distplot(train_df['identity_hate'],kde=False)

'''
# train_df.head()

comment_df = train_df['comment_text']

train_df['frequency'] = comment_df.apply(lambda x: len(x.split()))

train_df.info()

comment_df_test = test_df['comment_text']

test_df['frequency'] = comment_df_test.apply(lambda x: len(x.split()))

test_df.info()
test_df.describe()

sns.distplot(test_df['frequency'],bins=50)
t=Tokenizer(num_words=5000)

t.fit_on_texts(pd.concat([comment_df, comment_df_test]))

#print(t.word_index)

#print(comments)

#print(t.word_counts)

train_df.head()
X = t.texts_to_sequences(comment_df[0:])

X_test = t.texts_to_sequences(comment_df_test[0:])



Y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

from keras.preprocessing import sequence

max_length = 256

X = sequence.pad_sequences(X, maxlen = max_length)

X_test = sequence.pad_sequences(X_test, maxlen = max_length)



from keras.layers import Dropout

from keras.callbacks import ModelCheckpoint

embedding_vecor_length = 64

top_word = 5000

model = Sequential()

model.add(Embedding(top_word, embedding_vecor_length, input_length=max_length))

model.add(Dropout(0.2))

model.add(LSTM(128))

model.add(Dropout(0.2))

model.add(Dense(6, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''

print(model.summary())

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

model.fit(X, Y, epochs=3,callbacks=callbacks_list,validation_split=0.1, batch_size=64)

'''
#load weights

model.load_weights('../input/weights/weights.best.hdf5')

predictions=model.predict(X_test)

submissions=pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

submissions[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]=predictions

submissions.head()
submissions.to_csv('out.csv',index=False)