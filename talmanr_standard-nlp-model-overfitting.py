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
#  read the data
Data1 = pd.read_csv("../input/Sheet_1.csv",usecols=['response_id','class','response_text'],encoding='latin-1')
Data2 = pd.read_csv("../input/Sheet_2.csv",encoding='latin-1')
Data1.head()
Sampls = Data1['response_text']

# Delet panctioation marks, numbers and short
import nltk
AllWords=[]
Sentences = [ nltk.word_tokenize(s) for s in Sampls]
for i,words in enumerate(Sentences):
    words=[word.lower() for word in words if word.isalpha()]
    AllWords.append(words)
flat_list = [item for sublist in AllWords for item in sublist]


# tokenize create:
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(flat_list)

# transform the data:
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Conv1D
SeqData = tokenizer.texts_to_sequences(AllWords)
MatData = tokenizer.texts_to_matrix(Sampls.tolist())

All_input = sequence.pad_sequences(SeqData,maxlen=100)
All_output = np.array([x==  'flagged' for x in Data1['class'].tolist()],dtype=int)

# split into train and text
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( All_input, All_output, test_size=0.13, random_state=42)
# create NN 
from keras.models import Sequential
from keras.layers import Dense,Embedding,SimpleRNN,Dropout,LSTM
model = Sequential()
model.add(Embedding(1000,32))
model.add(LSTM(32))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(X_train,y_train,epochs=10,batch_size=8)
# predict results
Pred = model.predict_classes(X_test)
Accuracy = sum(Pred.flat == y_test)/len(y_test)
print(Accuracy)




