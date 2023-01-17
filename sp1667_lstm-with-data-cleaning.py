import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage

compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,

choice of loss and optimizer is critical, etc.

Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different

from what you see with CNNs/MLPs/etc.

'''
from __future__ import print_function

from keras.datasets import imdb

from keras.layers import Dense, Embedding

from keras.layers import LSTM

from keras.models import Sequential

from keras.preprocessing import sequence
train = pd.read_csv("../input/imdb-sentiment-classification/IMDB_Train.csv", low_memory = False, encoding='latin1')
train.review[0]
train.label[0]
train.review = train.review.str.lower()
from string import punctuation

print(punctuation)
def remove_punctuation(reviews):

    all_text = ''.join([c for c in reviews if c not in punctuation])

    return all_text



train['cleaned_review'] = train['review'].apply(remove_punctuation) 
from collections import Counter



all_text2 = ' '.join(list(train.cleaned_review))

# create a list of words

words = all_text2.split()

# Count all the words using Counter Method

count_words = Counter(words)



total_words = len(words)

sorted_words = count_words.most_common(total_words)
sorted_words
vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)}
len(vocab_to_int)
vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
#tokenize



reviews_int = []

for review in train.cleaned_review:

    r = [vocab_to_int[w] for w in review.split()]

    reviews_int.append(r)

#print (reviews_int[0:3])
encoded_labels = [1 if label =='pos' else 0 for label in train.label]

encoded_labels = np.array(encoded_labels)
set(train.label)
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

reviews_len = [len(x) for x in reviews_int]

pd.Series(reviews_len).hist()

plt.show()

pd.Series(reviews_len).describe()
reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]

encoded_labels = np.array([ encoded_labels[i] for i, l in enumerate(reviews_len) if l> 0 ])
max_features = len(vocab_to_int)#20000

# cut texts after this number of words (among top max_features most common words)

maxlen = 80

batch_size = 32
from keras.utils import to_categorical
print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(reviews_int, maxlen=maxlen)

y_train = to_categorical(encoded_labels,2)

print('x_train shape:', x_train.shape)

print('x_train shape:', y_train.shape)
reverse_dic = {y:x for x,y in vocab_to_int.items()}
print('Build model...')

model = Sequential()

#Size of the vocabulary, i.e. maximum integer index + 1.

model.add(Embedding(max_features+1, 128))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2, activation='softmax'))

# try using different optimizers and different optimizer configs

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
print('Train...')

model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=1, validation_split = 0.2)
x_train.shape
scores = model.predict(x_train)
#The model we trained outputs 2 numbers for each predction the 1st one is the probability that x belongs to class negative(0) sentiment

# The second number is the probability that x belongs to class positive(1) sentiment.

scores[0]
from sklearn.metrics import roc_auc_score
#Read more about the auroc scores

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html



#next we can predict teh auroc scores for our training dataset (just a demonstration - but there is little value outputing the training scores)



# we convert our ground truth array which had 2 values to an array with a sungle value

ground_truth = np.argmax(y_train, axis = 1)



#We only collect the scores for the postive class by selecting the second column of our scores array

prediction_class_positive = scores[:,1]
ground_truth
roc_auc_score(ground_truth,prediction_class_positive)
test = pd.read_csv("../input/imdb-sentiment-classification/IMDB_Test.csv")
"""

Insert Code here to clean your test set



"""
def create_scoring_df(df):

    df['id'] = df.index

    df = df[['id','label']]

    return df
output = create_scoring_df(test)
#The output dataset shares the same index as your test dataset.

#After you create clean your test dataset, get the "scores" like shown before on the test dataset.

#Then replace the "label" column values of the output dataset with the scores i.e. output['label'] = scores

#This should allow you to submit an entry.