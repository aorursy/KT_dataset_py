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
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
plt.style.use("ggplot")


import tensorflow as tf
data = pd.read_csv("/kaggle/input/entity-annotated-corpus/ner_dataset.csv", encoding = "ISO-8859-1", error_bad_lines=False)
data.head()
data.isna().sum()
data = data.fillna(method='ffill')
data.isna().sum()
data.head(20)
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words = list(set(data['Word'].values))
words.append("ENDPAD")
num_words = len(words)
tags = list(set(data['Tag'].values))
num_tags = len(tags)
num_words, num_tags
### Retrieve Sentences and correspondin Tags
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        agg_func = lambda s:[(w,p,t) for w, p, t in zip(s["Word"].values.tolist(),
                                                       s["POS"].values.tolist(),
                                                       s["Tag"].values.tolist())]
        
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
                
getter = SentenceGetter(data)
sentences = getter.sentences
sentences[0]
sentences[1]
### Define Mapping between Sentences and Tags
word2idx = {w: i+1 for i, w in enumerate(words)}
tag2idx = {t: i for i,t, in enumerate(tags)}
word2idx
tag2idx
### Padding Input sentences and creating train and test split
### Checking distribution of sentences length
plt.hist([len(s) for s in sentences], bins=50)
plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences = X, padding='post', value=num_words-1)

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences = y, padding='post', value= tag2idx["O"])
y = [to_categorical(i, num_classes=num_tags) for i in y]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state= 1)
### Build and Compile a Bidirectional LSTM Model
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional
input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words,output_dim=max_len,input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100,return_sequences=True,recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_tags,activation='softmax'))(model)
model = Model(input_word,out)
model.summary()
model.compile(optimizer = 'adam',
              loss='categorical_crossentropy',
             metrics=['accuracy'])
#### Train the model 
!pip install livelossplot
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback
import gc
gc.collect(),gc.collect()
early_stopping = EarlyStopping(monitor='val_accuracy',patience=1, verbose=0, mode='max',restore_best_weights= False)
callbacks = [PlotLossesCallback(),early_stopping]

history = model.fit(X_train,np.array(y_train),validation_split = 0.2,
                   batch_size = 32, epochs = 3, verbose=1,callbacks=callbacks)
### Evaluate Named Entity Model
model.evaluate(X_test, np.array(y_test))
i = np.random.randint(0, X_test.shape[0])
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)

y_true = np.argmax(np.array(y_test), axis=1)[i]
print("{:15}{:5}\t {}\n".format("Word","True","Pred"))
print("_"*30)
for W, true, pred in zip(X_test[i],y_true,p[0]):
    print("{:15}{}\t{}".format(words[W-1],tags[true],tags[pred]))

