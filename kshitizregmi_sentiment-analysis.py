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
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='latin-1')
df.head(3)
col = ['target','id','date','flag','user','text']
df.columns = col
df.drop(['id','date','flag','user'],axis = 1,inplace = True)
df.head()
df_n = df[(df['target'] ==4) | (df['target'] == 0) ]
df_n.target.unique()
df_n.target.replace({0: "NEGATIVE", 4: "POSITIVE"},inplace=True)
df_n.target.unique()
df_n.head(3)
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
import matplotlib.pyplot as plt
from collections import Counter

target_cnt = Counter(df_n.target)

plt.figure(figsize=(8,3))

plt.bar(target_cnt.keys(), target_cnt.values(),color='green')

plt.title("Dataset labels distribuition")
target_cnt
import re

df_n['text'] = df_n['text'].apply(lambda x: x.lower())

df_n['text'] = df_n['text'].apply((lambda x: re.sub(TEXT_CLEANING_RE,' ',x)))
df_n['text'][100:105]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import SGDClassifier

model = make_pipeline( CountVectorizer(),

                      TfidfTransformer(),

                      SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=100, random_state=42)

                     )

                    

df_n.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    df_n.text, df_n.target, test_size=0.5, random_state=42)
model.fit(X_train,y_train)
prediction = model.predict(X_test)
prediction[:5]
from sklearn.metrics import confusion_matrix
y_test.shape
confusion_matrix = confusion_matrix(y_test, prediction)

confusion_matrix
list1 = ["Actual Negative", "Actual Positive"]

list2 = ["Predicted Negative", "Predicted Positive"]

pd.DataFrame(confusion_matrix, list1,list2)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.utils.np_utils import to_categorical
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, split=' ')

tokenizer.fit_on_texts(df_n['text'].values)

X = tokenizer.texts_to_sequences(df_n['text'].values)
X = pad_sequences(X)
Y = pd.get_dummies(df_n['target']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.5, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
embed_dim = 128

lstm_out = 196

max_features = 2000



from keras.layers import SpatialDropout1D

model = Sequential()

model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2,activation='softmax'))

print(model.summary())
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs = 5, batch_size=32, verbose = 2)