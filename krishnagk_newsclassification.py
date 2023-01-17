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
!pip install seaborn

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        data = pd.read_csv(filename)

data = pd.read_csv('/kaggle/input/newsclassificationcsv/Copy of NewsClassificationOne - Sheet1.csv')

sns.countplot(data.Topic)

plt.xlabel('Label')

plt.title('Number of news messages')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

%matplotlib inline

import re



data.loc[:,'Description'] = data.Description.apply(lambda x : str.lower(x))





data.loc[:,'Description'] = data.Description.apply(lambda x : " ".join(re.findall('[\w]+',x)))



!pip install stop-words

from stop_words import get_stop_words

stop_words = get_stop_words('en')



def remove_stopWords(s):

    '''For removing stop words

    '''

    s = ' '.join(word for word in s.split() if word not in stop_words)

    return s



data.loc[:,'Description'] = data.Description.apply(lambda x: remove_stopWords(x))

X = data.Description

Y = data.Topic

le = LabelEncoder()

Y = le.fit_transform(Y)

Y = Y.reshape(-1,1)
data.head(15)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.10)



max_words = 5000

max_len = 150

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
def RNN():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model





model = RNN()

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(sequences_matrix,Y_train,batch_size=10,epochs=200,

          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))