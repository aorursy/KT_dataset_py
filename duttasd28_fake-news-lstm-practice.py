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
#Read the data

df = pd.read_csv('/kaggle/input/fake-news/train.csv')

df.head()
df = df.dropna()
X = df.drop('label', axis=1)

y = df['label']

X.shape, y.shape
import tensorflow as tf
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense
### Define vocabulary size

voc_size = 5000

messages = X.copy()

messages.reset_index(inplace=True)
### Data Preprocessing

# Remove stopwords

import nltk

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

import re

from nltk.corpus import stopwords

nltk.download('stopwords')
corpus = []

for i in range(len(messages)):

    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])

    review = review.lower()

    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)



corpus[:5]
# One hot representation

onehot = [one_hot(words, voc_size) for words in corpus]

onehot[:5]
max([len(vec) for vec in onehot])
# Pad the sentences, make fixed length

max_length = 50

embedded_docs = pad_sequences(onehot, padding = 'pre', maxlen = max_length)

embedded_docs[:5]
embedding_features_length = 40

from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(Embedding(voc_size, embedding_features_length, input_length = max_length))

model.add(Dropout(0.4))

model.add(LSTM(100))

model.add(Dropout(0.4))

model.add(Dense(1, activation = 'sigmoid'))
#Compile the model

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())

import numpy as np

X_final = np.array(embedded_docs)

y_final = np.array(y)
X_final.shape, y_final.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.2, random_state = 40)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Fit the model

# Now we fit the model

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 5, batch_size = 64)

y_pred = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)