# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





import keras

import sklearn

from tqdm import tqdm

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import LSTM,Dense,Embedding,Dropout,CuDNNGRU

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
print(df['target'].value_counts())

sns.countplot(df['target'])
embedding_vector = {}

f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt') 

for line in tqdm(f):

    vector = line.split(' ')

    word = vector[0]

    coef = np.asarray(vector[1:],dtype = 'float32')

    embedding_vector[word]=coef

f.close()

print('Number of words found ',len(embedding_vector))
x = df['question_text']

y = df['target']
token = Tokenizer()

token.fit_on_texts(x)
sequence = token.texts_to_sequences(x)
pad_seq = pad_sequences(sequence,maxlen = 100)
vocab_size = len(token.word_index)+1
embedding_matrix = np.zeros((vocab_size,300))

for word,i in tqdm(token.word_index.items()):

    embedding_vectors = embedding_vector.get(word)

    if embedding_vectors is not None:

        embedding_matrix[i] = embedding_vector[word]
model = Sequential()
model.add(Embedding(vocab_size,300,weights = [embedding_matrix],input_length =100,trainable = False))
model.add(CuDNNGRU(64))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])
history = model.fit(pad_seq,y,epochs = 5,batch_size=32,validation_split=0.2)
values = history.history

val_loss = values['val_loss']

training_loss = values['loss']

training_acc = values['acc']

validation_acc = values['val_acc']

epochs = range(5)



plt.plot(epochs,val_loss,label = 'Validation Loss')

plt.plot(epochs,training_loss,label = 'Training Loss')

plt.title('Epochs vs Loss')

plt.legend()

plt.show()
plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.title('Epochs vs Accuracy')

plt.legend()

plt.show()
testing = pd.read_csv('../input/test.csv')

testing.head()
x_test = testing['question_text']
x_test = token.texts_to_sequences(x_test)
testing_seq = pad_sequences(x_test,maxlen=100)
predict = model.predict_classes(testing_seq)
testing['label'] = predict
testing.head()
submit_df = pd.DataFrame({"qid": testing["qid"], "prediction": testing['label']})

submit_df.to_csv("submission.csv", index=False)