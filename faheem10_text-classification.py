# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
pos_files = os.listdir("../input/aclimdb/aclImdb/train/pos")

pos_files = ["../input/aclimdb/aclImdb/train/pos/"+file for file in pos_files]

pos_labels = [1 for _ in range(len(pos_files))]

neg_files = os.listdir("../input/aclimdb/aclImdb/train/neg")

neg_files = ["../input/aclimdb/aclImdb/train/neg/"+file for file in neg_files]

neg_labels = [0 for _ in range(len(neg_files))]



print(len(pos_files))

print(len(pos_labels))

print(len(neg_files))

print(len(neg_labels))
files = pos_files + neg_files

labels = pos_labels + neg_labels



print(len(files))

print(len(labels))
print(files[:2])
from sklearn.model_selection import train_test_split
file_train, file_test, labels_train, labels_test = train_test_split(files, labels,

                                                                   test_size = 0.2,

                                                                   shuffle = True,

                                                                   random_state = 0)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
bow = TfidfVectorizer('filename', tokenizer = word_tokenize, stop_words = set(stopwords.words('english')))
X_train = bow.fit_transform(file_train)
X_train.toarray().shape
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train, labels_train)
X_test = bow.transform(file_test)
clf.score(X_test.toarray(), labels_test)
review = ['it was a good movie']
X_pred = ["../input/aclimdb/aclImdb/test/pos/"+file for file in os.listdir("../input/aclimdb/aclImdb/test/pos")][0]
X_pred = bow.transform([X_pred])
clf.predict(X_pred)
import numpy as np
print(X_pred)
open(X_pred, 'r').readlines()
text = f.readlines()[0]
print(text)
import re
' '.join(re.split('\W+', text))
print(len(files))
reviews = [' '.join(re.split('\W+', open(file, 'r').readlines()[0])) for file in files]
import pandas as pd
df = pd.DataFrame({'reviews': reviews, 'labels': labels})

df.head()
df.labels.value_counts()
df.reviews[0]
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.reviews.values)
df.reviews[0]
df.reviews = tokenizer.texts_to_sequences(df.reviews)
len(tokenizer.word_index)
df.reviews.head()
max_len = max([len(review) for review in df.reviews])

print(max_len)
type(df.reviews.values)
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
labels = to_categorical(df.labels.values)
# reviews = pad_sequences(sequences=df.reviews, maxlen = max_len)

# reviews = pad_sequences(sequences=df.reviews, maxlen = max_len)
type(reviews)
from keras.models import Model

from keras.layers import Embedding, Dense, LSTM, Input
inp = Input(shape=(None,))



x = Embedding(74743, 128)(inp)

x = LSTM(512)(x)

out = Dense(2, activation = 'softmax')(x)
model = Model(inp, out)
model.summary()
model.compile(loss = 'categorical_crossentropy',

             optimizer = 'adam',

             metrics = ['accuracy'])
model.fit(np.asarray(df.reviews), labels, batch_size = 32,

         epochs = 20, validation_split = 0.2)