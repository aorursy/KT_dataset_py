# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df
df['text'] = [x.replace(',', ' , ') for x in df['text']]

df['text'] = [x.replace("'", " ' ") for x in df['text']]

df['text'] = [x.replace(".", " . ") for x in df['text']]

df
import gensim

sentences = [x.split() for x in df['text']]

model_w2v = gensim.models.Word2Vec(sentences, min_count=1)
#sentences[:3]

print(model_w2v.wv.most_similar(positive=['heart']))

print(len(model_w2v.wv['heart']))
df['words'] = [x.split() for x in df['text']]

w2v = []

zerovec = 100*[0]

max_length = 256



for x in df['words']:

    vec = []

    #print(len(x))

    for w in x[:max_length]:

        vec.extend(model_w2v.wv[w])

    #print(len(vec))

    remaining = max_length*100 - len(vec)

    if remaining > 0 :

        padding = remaining*[0]

        vec.extend(padding)

        w2v.append(vec)

    elif remaining > 0 :

        w2v.append(vec[:max_length])

    else :

        w2v.append(vec)
print(len(w2v[0]))
from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding

#vocab_size = 500

#df['encoded_text'] = [one_hot(x, vocab_size) for x in df['text']]

print(len(w2v))

print(len(w2v[0]))
# pad documents to a max length of 4 words

#max_length = 256000

#padded_text = pad_sequences(w2v, maxlen=max_length, padding='post')

class PaddedText2Vec(object):

    def __init__(self, t2v, maxlen):

        self.t2v = t2v

        self.maxlen = maxlen

 

    def __iter__(self):

        for v in self.t2v:

            remaining = self.maxlen - len(v)

            if remaining > 0 :

                padding = remaining*[0]

                v.extend(padding)

                yield v

            elif remaining > 0 :

                yield v[:self.maxlen]

            else :

                yield v
#paddedt2v = PaddedText2Vec(w2v, 256000)
#for x in paddedt2v:

#    print(len(x))

#    break
from keras import regularizers, optimizers

from keras.layers import BatchNormalization

# define the model

max_length = 256

model = Sequential()

model.add(Dense(150, input_shape=(max_length*100,)))

#model.add(Embedding(vocab_size, 100, input_length=max_length))

#model.add(Flatten())

#model.add(BatchNormalization())

model.add(Dense(150, activation='elu', kernel_regularizer=regularizers.l2(0.01)))

#model.add(Dense(200, activation='elu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

# compile the model

sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())
import keras

from keras.layers import Dense, GlobalAveragePooling1D, Embedding

import keras.backend as K

from keras.callbacks import EarlyStopping

from keras.models import Sequential



from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

a2c = {'EAP':0, 'HPL':1, 'MWS':2}

labels = to_categorical([a2c[x] for x in df['author']])
labels[:10]
# fit the model

model.fit(w2v, labels, epochs=20, verbose=1, batch_size=128, validation_split=0.15)
df_test = pd.read_csv('../input/test.csv')

df_test['words'] = [x.split() for x in df_test['text']]

w2v_test = []

zerovec = 100*[0]

max_length = 256



for x in df_test['words']:

    vec = []

    #print(len(x))

    for w in x[:max_length]:

        try:

            vec.extend(model_w2v.wv[w])

        except:

            continue

    #print(len(vec))

    remaining = max_length*100 - len(vec)

    if remaining > 0 :

        padding = remaining*[0]

        vec.extend(padding)

        w2v_test.append(vec)

    elif remaining > 0 :

        w2v_test.append(vec[:max_length])

    else :

        w2v_test.append(vec)
y_pred = model.predict_proba(w2v_test)



result = pd.read_csv('../input/sample_submission.csv')

for a, i in a2c.items():

    result[a] = y_pred[:, i]
result.to_csv('out.csv', index=False)
print(one_hot('man', 2))

print(one_hot('woman', 2))
