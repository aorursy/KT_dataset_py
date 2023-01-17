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
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import pandas as pd

import string



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from keras.datasets import imdb

from keras.preprocessing import sequence

from keras.layers import Dense

from keras.models import Sequential

from keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dropout

from keras.preprocessing import sequence

from keras.optimizers import Adam

import keras



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
table = str.maketrans('', '', string.punctuation)
inputdir = '/kaggle/input/nlp-getting-started/'

outputdir = '/kaggle/working/'
fname_train = os.path.join(inputdir, 'train.csv')

fname_test = os.path.join(inputdir, 'test.csv')

train = pd.read_csv(fname_train, encoding='UTF8')

test = pd.read_csv(fname_test, encoding='UTF8')



train['text'] = train['text'].str.lower()

test['text'] = test['text'].str.lower()
new_text = []

for words in train['text']:

    words = words.split()

    stripped = [w.translate(table) for w in words]

    new_text.append(' '.join(stripped))

    

train['new_text'] = new_text
X = train['new_text']

y = train['target']
def create_index(X_train):

    words = set()



    for s in X_train:

        for w in s.split():

            words.add(w)



    word2index = {w: i + 2 for i, w in enumerate(list(words))}

    word2index['#PAD#'] = 0

    word2index['#OOV#'] = 1



    return word2index
word2index = create_index(X)
def data_create(X, word2index):

    new_X = []

    for s in X:

        s_int = []

        for w in s.split():

            s_int.append(word2index.get(w,0))

        new_X.append(s_int)



    return new_X
new_X = data_create(X, word2index)

len(new_X)
lengths = [len(s) for s in new_X]

OPT_LENGTH = int(np.average(lengths) + (2 * np.std(lengths))) 

MAX_LENGTH = np.max(lengths)

print(OPT_LENGTH, MAX_LENGTH)
# plot the distribution to make sure that OPT_LENGTH is good for network training



sns.distplot(lengths);



# as you can see in the plot the OPT_LENGTH of 26 (=avg + 2*stdev) will cover > 90% data. So, it is good to pad data points 

# for our network
new_X = sequence.pad_sequences(new_X, maxlen= OPT_LENGTH, padding='post')

print(new_X.shape, y.shape)
pd.Series(y).value_counts()
callbacks_list = [

            keras.callbacks.EarlyStopping(

                monitor='val_acc',

                patience=1

            )

        ]
# Hyperparameters



no_epochs = 10

batch_size = 8

no_neurons = 32

val_split = 0.1
model = Sequential()

model.add(Embedding(len(word2index), 16))

model.add(Bidirectional(LSTM(no_neurons, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(new_X, y,

                    epochs=no_epochs,

                    batch_size=batch_size,

                    callbacks = callbacks_list,

                    validation_split=val_split)
# Prediction using Train data



y_pred_model = model.predict_classes(new_X)
# Confusion matrix

confusion_matrix(y, y_pred_model)
print(f"F1 Score: {f1_score(y, y_pred_model)}")

print(f"Precision Score: {precision_score(y, y_pred_model)}")

print(f"Recall Score: {recall_score(y, y_pred_model)}")
new_test_text = []

for words in test['text']:

    words = words.split()

    stripped = [w.translate(table) for w in words]

    new_test_text.append(' '.join(stripped))

    

test['new_text'] = new_test_text 
X_test = test['new_text']
print(X_test.shape, test.shape)
new_X_test = data_create(X_test, word2index)

new_X_test = sequence.pad_sequences(new_X_test, maxlen= OPT_LENGTH, padding='post')

print(len(new_X_test), new_X_test.shape)
y_pred_test = model.predict_classes(new_X_test).reshape(-1,)

y_pred_test.shape
dict1 = {'id': list(test['id']), 'test_data':list(X_test), 'target':list(y_pred_test)}

test_df = pd.DataFrame(dict1)
# quality of predicted target value

test_df['target'].value_counts()
out_pred_file = os.path.join(outputdir, 'test_prediction.csv')

print(f"Test data prediction file: {out_pred_file}")
# writing to test final prediction file



test_df[['id', 'target']].to_csv(out_pred_file,index=False)

print('output file saved')