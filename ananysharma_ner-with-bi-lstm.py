# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/entity-annotated-corpus/ner_dataset.csv",encoding = 'latin1')
data = data.fillna(method = 'ffill')
data.head()
# data.shape
data.nunique()
words = list(set(data["Word"].values))
words.append("ENDPAD")
num_words = len(words)
num_words
words_tag = list(set(data["Tag"].values))
# words_tag.append("ENDPAD")
num_words_tag = len(words_tag)
num_words_tag
num_words,num_words_tag
group = data.groupby(data["Sentence #"])
# group.groups
class Get_sentence(object):
    def __init__(self,data):
        self.n_sent=1
        self.data = data
        agg_func = lambda s:[(w,p,t) for w,p,t in zip(s["Word"].values.tolist(),
                                                     s["POS"].values.tolist(),
                                                     s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
getter = Get_sentence(data)
sentence = getter.sentences
sentence[0]
word_idx = {w : i+1 for i ,w in enumerate(words)}
tag_idx =  {t : i for i ,t in enumerate(words_tag)}
plt.hist([len(s) for s in sentence],bins= 50)
plt.xlabel("Length of Sentences")
plt.show()
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_len = 50
X = [[word_idx[w[0]] for w in s] for s in sentence]
X = pad_sequences(maxlen = max_len,sequences = X,padding = 'post',value = num_words-1)
y = [[tag_idx[w[2]] for w in s] for s in sentence]
y = pad_sequences(maxlen = max_len,sequences = y,padding = 'post',value = tag_idx['O'])
y = [to_categorical(i,num_classes = num_words_tag) for i in  y]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state=1)
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional
input_word = Input(shape = (max_len,))
model = Embedding(input_dim = num_words,output_dim = max_len,input_length = max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100,return_sequences = True, recurrent_dropout = 0.1))(model)
out = TimeDistributed(Dense(num_words_tag,activation = 'softmax'))(model)
model = Model(input_word,out)
model.summary()
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
# from livelossplot import PlotLossesKeras
early_stopping = EarlyStopping(monitor = 'val_accuracy',patience =2,verbose = 0,mode = 'max',restore_best_weights = False)
callbacks = [early_stopping]

history = model.fit(
    x_train,np.array(y_train),
    validation_split =0.2,
    batch_size = 64,
    epochs = 3,
    verbose =1
)



model.evaluate(x_test,np.array(y_test))
i = np.random.randint(0, x_test.shape[0])
p = model.predict(np.array([x_test[i]]))
# print(np.shape(p))
# print(p)
p = np.argmax(p, axis=-1)


y_true = np.argmax(np.array(y_test), axis=-1)[i]

print("{:15}{:5}\t{}\n".format("Word", "True", "Pred"))
print("-"*30)

for (w, t, pred) in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], words_tag[t], words_tag[pred]))

