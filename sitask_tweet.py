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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from matplotlib import pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords

import re

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import SpatialDropout1D

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping



from sklearn.model_selection import train_test_split
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
train_df[train_df['text'].isna()]
#let us remove this

train_df = train_df.drop(314)
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))



def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    #print(text)

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.

    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text

    return text
train_df['text'] = train_df['text'].apply(clean_text)

train_df['selected_text'] = train_df['selected_text'].apply(clean_text)
train_df['text_len'] = train_df['text'].str.split().str.len()

train_df['seltext_len'] = train_df['selected_text'].str.split().str.len()

train_df['diff_len'] = train_df['text_len'] - train_df['seltext_len']
df = train_df['diff_len'].value_counts().reset_index()
df.head()
df.columns = ['Diff Length', 'Count']
plt.figure(figsize=(15,15))

sns.barplot(x=df['Diff Length'], y=df['Count'])
train_df.groupby(['diff_len','sentiment']).count()
train_df['sentiment'].value_counts()
test_df['sentiment'].value_counts()
pos_train = train_df[train_df['sentiment'] == 'positive']

neutral_train = train_df[train_df['sentiment'] == 'neutral']

neg_train = train_df[train_df['sentiment'] == 'negative']
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 100000

# Max number of words in each tweet

MAX_SEQUENCE_LENGTH = 30

# This is fixed.

EMBEDDING_DIM = 500

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

words = np.concatenate((train_df['text'], test_df['text']))

tokenizer.fit_on_texts(words)#train_df['text'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(pos_train['text'].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
Y = tokenizer.texts_to_sequences(pos_train['selected_text'].values)

Y = pad_sequences(Y, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', Y.shape)
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.2, random_state = 17)

print(X_train.shape,Y_valid.shape)

print(X_valid.shape,Y_valid.shape)
model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(30, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



epochs = 5

batch_size = 64



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
pos_test = test_df[test_df['sentiment'] == 'positive']

neutral_test = test_df[test_df['sentiment'] == 'neutral']

neg_test = test_df[test_df['sentiment'] == 'negative']
Z = tokenizer.texts_to_sequences(pos_test['text'].values)

#Z = tokenizer.texts_to_sequences(test_df['text'].values)

Z = pad_sequences(Z, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', Z.shape)
p = model.predict(Z)
p = p>1e-1
idx_word = tokenizer.index_word
def build_text_string(j, Z):

    s = ' '

    #print(p[j])

    for i in range(30):   

        if p[j][i]!=0:

            #print(p[j][i])

            if( Z[j][i] != 0):

                s += idx_word[Z[j][i]]

                s += ' '

    return( s )

        
new_pos_test = pos_test.copy().reset_index(drop=True)
for j in range(len(Z)-1) :

    #print(j)

    s = build_text_string(j, Z)

    #print('sel text = ', s)

    new_pos_test.loc[j, 'selected_text'] = s

    #print('text = ', new_pos_test.loc[j, 'text'])
new_pos_test.head()
Zn = tokenizer.texts_to_sequences(neg_test['text'].values)

Zn = pad_sequences(Zn, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', Zn.shape)
n = model.predict(Zn)
n = n>1e-3
new_neg_test = neg_test.copy().reset_index(drop=True)
for j in range(len(Zn)-1) :

    #print(j)

    s = build_text_string(j, Zn)

    #print('sel text = ', s)

    new_neg_test.loc[j, 'selected_text'] = s

    #print('text = ', new_neg_test.loc[j, 'text'])

    #print(j)
neutral_test['selected_text'] = neutral_test['text']
frames = [new_neg_test, new_pos_test, neutral_test]

merged = pd.concat( frames )
#merged = merged.merge(neutral_test)
merged.head()
s = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
s.head()
final = pd.merge( s, merged, how='left', on='textID')
final.head()
final = final.drop(columns=['text_x', 'text_y', 'sentiment_x', 'sentiment_y'])
final.head()
#final.columns = ['textID', 'selected_text']
import csv

#final.to_csv('submission.csv',quoting = csv.QUOTE_NONE,quotechar="",escapechar = ',',index=False)

final.to_csv('submission.csv', index=False)