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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords 

df = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
df.shape
df.columns
df.describe()
df.info()
df['Text'] =df["Text"].fillna("")
df['Summary'] =df['Summary'].fillna("")
df.info()
df = df.drop(['Time', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator'], axis = 1)
df['Score'] = np.where(df['Score']<3, 0, df['Score'])
df['Score'] = np.where(df['Score']!=0, 1, df['Score'])
df['Score']
df = df.rename(columns = {'Score':'Target'})
df
X_sum = df.iloc[:, [1, 2]].values
X_text = df.iloc[:, [1,3]].values
y = df.iloc[:, 1].values
text = X_sum[:, 1]
sw = list(set(stopwords.words('english')))
wlist = []
for i in text:
    wlist.append(i.split(' '))

wlist
twords = []
for i in wlist:
    liste = [j for j in i if not j in sw]
    twords.append(liste)
p = []
for i in twords:
    for j in i:
        p.append(j)
#vocab =len(set(p))
final = []
for i in range(0, len(twords)):
     final.append(' '.join(j for j in twords[i]))
final
training = []
for i in range(0, len(final)):
    listing =[]
    listing.append(final[i])
    training.append(listing)
training
tokenizer = Tokenizer()
from sklearn.model_selection import train_test_split
X_training, X_testing, y_training, y_testing = train_test_split(training, y, test_size = 0.2)
tokenizer.fit_on_texts(X_training)
preprocessed= tokenizer.texts_to_sequences(X_training)
preprocessed = pad_sequences(preprocessed, maxlen = 2000, padding = 'post')
preprocessed[-1]
preprocessed.shape
maxims = []
for i in preprocessed:
    maxims.append(max(i))
vocab = max(maxims)
vocab
from keras.models import Sequential
model = Sequential()

from keras.layers import Embedding, LSTM, Dropout, Dense, GRU

model.add(Embedding(vocab+1, 256,trainable=True, input_length = 2000))
model.add(Dropout(rate = 0.25))
model.add(LSTM(256, return_sequences=True, dropout=0.2))
model.add(GRU(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.15))
model.add(Dense(64))
model.add(Dropout(rate=0.25))
model.add(Dense(16))
model.add(Dropout(rate = 0.25))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(preprocessed, y_training, epochs = 2, batch_size = 20)