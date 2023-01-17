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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D, Bidirectional



import tensorflow as tf

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
import pandas as pd 



train = pd.read_csv('/kaggle/input/fake-news/train.csv')

test = pd.read_csv('/kaggle/input/fake-news/test.csv')



train = train.set_index('id', drop=True)

test = test.set_index('id', drop=True)
train[['title', 'author']] = train[['title', 'author']].fillna(value='missing')

train=train.dropna()

train.isnull().sum()
length = []

[length.append(len(str(text))) for text in train['text']]

train['length'] = length

train.head()
train = train.drop(train['text'][train['length'] < 50].index, axis = 0)
train.shape
length = []

[length.append(len(str(text))) for text in test['text']]

test['length'] = length

test.head()
min(train['length']), max(train['length']), round(sum(train['length'])/len(train['length']))
min(test['length']), max(test['length']), round(sum(test['length'])/len(test['length']))
max_features = 4500
tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ', oov_token='<OOV>')

tokenizer.fit_on_texts(texts = train['text'])

X = tokenizer.texts_to_sequences(texts = train['text'])

X = pad_sequences(sequences = X, maxlen = max_features, padding = 'post')

X
print(X.shape)

y = train['label'].values

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(

    X, 

    y, 

    test_size = 0.2, random_state = 101)
y_train
X_train
X_test
y_test


model = Sequential([

                    Embedding(input_dim=max_features, output_dim=120),

                    Bidirectional(LSTM(units=120, 

                         activation='tanh',

                         recurrent_activation='sigmoid',

                         unroll=False,

                         use_bias=True,

                         dropout = 0.2, 

                         recurrent_dropout = 0

                         )),

                    # Dropout(rate = 0.5),

                    Dense(units = 120,  activation = 'relu'),

                    # Dropout(rate = 0.5),

                    Dense(units = len(set(y)),  activation = 'sigmoid', name = 'output_layer')

])





model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
with tf.device('/device:GPU:0'):

  model_fit = model.fit(

      X_train, y_train, epochs = 1, verbose=1)
from sklearn.metrics import accuracy_score

prediction = model.predict_classes(

    X_test

)

accuracy_score(y_test, prediction)
test[['title', 'author']] = test[['title', 'author']].fillna(value='missing')

test = test.fillna(' ')
test.shape
test_sequences = tokenizer.texts_to_sequences(test['text'])

test_sequences = pad_sequences(sequences = test_sequences, maxlen = max_features, padding = 'post')

test_sequences
type(test['text'])
test_prediction = model.predict_classes(

    test_sequences

)
submission = pd.DataFrame({'id':test.index, 'label':test_prediction})

submission.shape
submission.head()
submission.to_csv('submission.csv', index=False)