# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Embedding

# define sequences

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import re

import os

import re

import nltk

import keras

#import math

#nltk.download('stopwords')

from nltk.corpus import stopwords

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/consumer_complaints.csv')

df.loc[df['product'] == 'Credit reporting', 'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'

df.loc[df['product'] == 'Credit card', 'product'] = 'Credit card or prepaid card'

df.loc[df['product'] == 'Payday loan', 'product'] = 'Payday loan, title loan, or personal loan'

df.loc[df['product'] == 'Virtual currency', 'product'] = 'Money transfer, virtual currency, or money service'

df = df.loc[df['product'] != 'Other financial service']





print(df)
dff=df['product'].value_counts().sort_values(ascending=False)

d=dff.to_dict()

#print(dff)

#py.iplot(dff,kind='bar', yTitle='Number of Complaints',title='Number complaints in each product')

pd.DataFrame(dff).plot(kind='bar',title='Number of complaints in each product')

#plt.show()



def print_plot(index):

    example = df[df.index == index][['consumer_complaint_narrative', 'product']].values[0]

    if len(example) > 0:

        print(example[0])

        print('Product:', example[1])

print_plot(256589)
df = df.reset_index(drop=True)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))

FLOAT=re.compile('[-+]?\d*\.\d+|\d+')



def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = FLOAT.sub('', text)

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.

    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    

    text = text.replace('x', '')

#    text = re.sub(r'\W+', '', text)

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text

    return text

#print(df['consumer_complaint_narrative'])

#dff=df

dff = pd.DataFrame(columns=df.columns)

#cond = df.consumer_complaint_narrative !='nan' and df.index<256599 and df.index>256589

#rows = df.loc[cond,:]

rows = df.loc[256589:266589,:]

dff = dff.append(rows, ignore_index=True)

for i,txt in dff['consumer_complaint_narrative'].iteritems():

#for i in range(len(dff)):

    #print(df.loc[i, "consumer_complaint_narrative"])

    #print(txt)|

    if isinstance(txt, str):

        if txt!='nan':

            tx=clean_text(txt)

            tx=tx.replace('\d+', '')

            dff.loc[dff.index[i], 'consumer_complaint_narrative']=tx

            #print(tx)

    else:

         dff.loc[dff.index[i], 'consumer_complaint_narrative']='nan'

    

    

#df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].apply(clean_text)

#df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].str.replace('\d+', '')

dff.to_csv("Clean.csv", sep='\t', encoding='utf-8')

print('Success')
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 250

# This is fixed.

EMBEDDING_DIM = 100

tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(dff['consumer_complaint_narrative'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(dff['consumer_complaint_narrative'].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(dff['product']).values

print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model = keras.models.Sequential()

model.add(keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(keras.layers.SpatialDropout1D(0.2))

model.add(keras.layers.recurrent.LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



epochs = 8

batch_size = 64



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1

                    ,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show();

plt.title('Accuracy')

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.legend()

plt.show();
new_complaint = ['I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account do not belong to me : XXXX.']

seq = tokenizer.texts_to_sequences(new_complaint)

padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

pred = model.predict(padded)

labels = ['Credit reporting, credit repair services, or other personal consumer reports', 'Debt collection', 'Mortgage', 'Credit card or prepaid card', 'Student loan', 'Bank account or service', 'Checking or savings account', 'Consumer Loan', 'Payday loan, title loan, or personal loan', 'Vehicle loan or lease', 'Money transfer, virtual currency, or money service', 'Money transfers', 'Prepaid card']

print(pred, labels[np.argmax(pred)])