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
data = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip",sep='\t')

data.head(10)
data.shape
data.info()
x_train = []

x_train = data['Phrase']

print(type(x_train))

x_train = np.asarray(x_train)

print(type(x_train))

x_train.shape
y_train=[]

y_train = data['Sentiment']

y_train = np.asarray(y_train)

print(type(y_train))

y_train.shape
x_train = [rev.replace(",","").replace(".","").lower() for rev in x_train]
(x_train[0])
Y_train = np.zeros((y_train.size, y_train.max()+1))

Y_train[np.arange(y_train.size),y_train] = 1

Y_train
Y_train.shape
review_max_len = 200

vocab_size = 5000
from keras.preprocessing.text import one_hot



x_train_num = [one_hot(i,vocab_size) for i in x_train]
from keras.preprocessing import sequence



X_train = sequence.pad_sequences(x_train_num,maxlen=review_max_len)

X_train[0]
X_train.shape
from keras.models import Sequential

from keras.layers import LSTM,Dense,Conv1D,MaxPool1D

from keras.layers.embeddings import Embedding
model = Sequential()



model.add(Embedding(vocab_size,

                   32,

                   input_length=review_max_len))



model.add(Conv1D(32,(3),activation='relu'))

model.add(MaxPool1D(2))

model.add(LSTM(80,dropout=0.25,recurrent_dropout=0.25))

model.add(Dense(5,activation='softmax'))



model.compile(optimizer='adam',

             loss="categorical_crossentropy",

             metrics=['accuracy'])



model.summary()
model.fit(X_train,Y_train,

         batch_size=256,

         epochs=10,

         verbose=2)
test = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip",sep="\t")

test
x_test = []

x_test = test['Phrase']

print(type(x_train))

x_test = np.asarray(x_test)

print(type(x_test))

x_test.shape
x_test = [rev.replace(",","").replace(".","").lower() for rev in x_test]
x_test_num = [one_hot(i,vocab_size) for i in x_test]

X_test = sequence.pad_sequences(x_test_num,maxlen=review_max_len)

X_test[0]
X_test.shape
predicts = model.predict(X_test,

                        batch_size=256,

                        verbose=2)
predicts.shape
preds_final = [np.argmax(i) for i in predicts]

print(len(preds_final))
out = pd.DataFrame(data=test.PhraseId,columns=['PhraseId'])

out
out['Sentiment'] = preds_final

out
out.to_csv('../working/submission.csv', index=False)