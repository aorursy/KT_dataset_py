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
from keras.models import Sequential

from keras.layers import CuDNNLSTM,Dense,Embedding,Dropout

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from tqdm import tqdm
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")

test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")

sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")
df = train[['Phrase','Sentiment']]
df.head()
f = open('../input/gloveembeddings/glove.6B.300d.txt')

embedding_values = {}

for line in tqdm(f):

    value = line.split(' ')

    word = value[0]

    coef = np.array(value[1:],dtype = 'float32')

    embedding_values[word]= coef

f.close()
token  = Tokenizer()
x = df['Phrase']

y = df['Sentiment']

y = to_categorical(y)
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)
pad_seq = pad_sequences(seq,maxlen=300)
vocab_size = len(token.word_index)+1

print(vocab_size)
embedding_matrix = np.zeros((vocab_size,300))

for word,i in tqdm(token.word_index.items()):

    values = embedding_values.get(word)

    if values is not None:

        embedding_matrix[i] = values
x_train,x_test,y_train,y_test = train_test_split(pad_seq,y,test_size = 0.3,random_state = 42)
model = Sequential()
model.add(Embedding(vocab_size,300,input_length=300,weights = [embedding_matrix],trainable = False))
model.add(CuDNNLSTM(75,return_sequences=True))

model.add(CuDNNLSTM(75,return_sequences=False))
model.add(Dense(128,activation = 'relu'))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=16,epochs = 5,validation_data=(x_test,y_test))
test.head()
test['Sentiment'] = ''
test.head()
testing_phrase = test['Phrase']
test_seq = token.texts_to_sequences(testing_phrase)
pad_test_seq = pad_sequences(test_seq,maxlen=300)
predict = model.predict_classes(pad_test_seq)
predict[0]
test['Sentiment']  = predict
test.head()
submission = test[['PhraseId','Sentiment']]
submission.head()
submission.to_csv('Submission.csv',index = False)