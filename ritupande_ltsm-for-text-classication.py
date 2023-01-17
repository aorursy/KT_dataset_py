# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_pd = pd.read_csv('../input/train.csv')
test_pd = pd.read_csv('../input/test.csv')
train_pd.head()
train_pd.isnull().any()
# extract the comments and labels 
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y_train = train_pd[list_classes].values
X_train_lst = train_pd["comment_text"].values
X_test_lst  = test_pd["comment_text"].values
X_test_ids  =  test_pd["id"].values
#set max dictionary size
dictionary_size = 20000
# tokenize the text 
tokenizer = Tokenizer(num_words=dictionary_size)
#create a dictionary index:word
tokenizer.fit_on_texts( X_train_lst )
#create Index representation of comments after tokenizing
X_train_tokenized_lst = tokenizer.texts_to_sequences(X_train_lst)
X_test_tokenized_lst  = tokenizer.texts_to_sequences(X_test_lst)
wordcount_per_comment = [len(comment) for comment in X_train_tokenized_lst]
plt.hist(wordcount_per_comment, bins = np.arange(0,800,10))
plt.show()    
# based on the hostogram most comments are less than 350 hence set max_comment_length = 200
max_comment_length = 200
X_train = pad_sequences(X_train_tokenized_lst, maxlen=max_comment_length)
X_test =  pad_sequences(X_test_tokenized_lst, maxlen=max_comment_length )

# create model

embedding_vecor_length = 60
model = Sequential()
model.add(Embedding(dictionary_size, embedding_vecor_length, input_length=max_comment_length))
model.add(LSTM(100))
model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

history = model.fit(X_train, Y_train, validation_split = 0.3, epochs=2, batch_size=64)
#prepare submission
predictions = model.predict(X_test, verbose=1)
print( predictions[0:5])
df1 = pd.DataFrame(data=predictions, columns= list_classes )
df1.head()


df2 = pd.DataFrame(X_test_ids, columns= ['id'] )
df2.head()


submission_df = pd.concat( [df2, df1], axis=1 )
submission_df.head()
submission_df.to_csv("toxic_comments_classification.csv",index=False)