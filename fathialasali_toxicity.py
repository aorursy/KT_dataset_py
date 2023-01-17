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
import pickle

import numpy as np

import pandas as pd

from types import SimpleNamespace



import tensorflow as tf

import tensorflow_hub as hub

import transformers



from tqdm.notebook import tqdm, trange

import matplotlib.pyplot as plt
tokenizer_type = transformers.DistilBertTokenizer

nlp_model_type = transformers.TFDistilBertForSequenceClassification

pretrained_model = 'distilbert-base-uncased'



input_layer_name = 'input'



num_epochs = 4

max_sequence_length = 256

batch_size = 256 # too large batch sizes cause memory errors
with open('/kaggle/input/toxicity/toxicity_data.pkl', 'rb') as f:

    data = pickle.load(f)

train = data[data.Training_evaluation_split.eq('Training')].drop(columns=['Source', 'Training_evaluation_split'])

test = data[data.Training_evaluation_split.eq('Evaluation')].drop(columns=['Source', 'Training_evaluation_split'])
train
train.shape
test.shape
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping
train.info()
plt.figure(figsize=(16, 8))

plt.title('Score distribution')

plt.xlabel('Toxicity (fraction of reviewers which labelled text as toxic)')

plt.ylabel('Number of comments')

plt.hist(train.Toxicity, bins=64)

plt.show()
import numpy as np

print("Hatred Toxicity: {}\nNon-hatred Toxicity: {}".format(

    (train.Toxicity == 1).sum(),

    (train.Toxicity == 0).sum()

))

hashtags = train['Text'].str.extractall('#(?P<hashtag>[a-zA-Z0-9_]+)').reset_index().groupby('level_0').agg(lambda x: ' '.join(x.values))

train.loc[:, 'hashtags'] = hashtags['hashtag']

train['hashtags'].fillna('', inplace=True)



train.loc[:, 'mentions'] = train['Text'].str.count('@[a-zA-Z0-9_]+')



train.Text = train.Text.str.replace('@[a-zA-Z0-9_]+', '')



"""Removing anything but the words"""



train.Text = train.Text.str.replace('[^a-zA-Z]', ' ')
from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

from nltk import pos_tag, FreqDist, word_tokenize



stemmer = SnowballStemmer('english')

lemmer = WordNetLemmatizer()



part = {

    'N' : 'n',

    'V' : 'v',

    'J' : 'a',

    'S' : 's',

    'R' : 'r'

}



def convert_tag(penn_tag):

    if penn_tag in part.keys():

        return part[penn_tag]

    else:

        return 'n'





def tag_and_lem(element):

    sent = pos_tag(word_tokenize(element))

    return ' '.join([lemmer.lemmatize(sent[k][0], convert_tag(sent[k][1][0]))

                    for k in range(len(sent))])

    



train.loc[:, 'Text'] = train['Text'].apply(lambda x: tag_and_lem(x))

train.loc[:, 'hashtags'] = train['hashtags'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
X = train.Text

Y = train.Toxicity

le = LabelEncoder()

Y = le.fit_transform(Y)

Y = Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)
max_words = 1000

max_len = 100

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
def RNN():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(64,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history = model.fit(sequences_matrix,Y_train,batch_size=64,epochs=8,

          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)



print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()