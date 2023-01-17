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
import numpy as np 

import pandas as pd

import random

from collections import Counter

import pprint

import time

import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays

import sys

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
train.head()
train.tail()
train.isnull().sum().sum()
print('Shapes')

print("TRAIN :",train.shape)

print(len(train[train["target"]==0]), " :  sincere questions")

print(len(train[train["target"]==1]), " :  insincere questions")
SAMPLE_SIZE = 80000

df_0 = train[train["target"]==0].sample(SAMPLE_SIZE, random_state = 101)

    # filter out class 1

df_1 = train[train["target"]==1].sample(SAMPLE_SIZE, random_state = 101)



df = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)

   # shuffle

df= shuffle(df)

df_data, df_test = train_test_split(df, test_size=0.2)



dictio = {0:"sincere question  ",1:"insincere question"}

def pretty_print_text_and_label(i):

    print(dictio[df_data.iloc[i][2]] + "\t:\t" + train.iloc[i][1][:80] + "...")
print("type \t : \t question\n")

# choose  a random spam set to analyse

# random.randrange(start, stop, step)

pretty_print_text_and_label(random.randrange(0,4572))

pretty_print_text_and_label(random.randrange(0,4572,4))

pretty_print_text_and_label(random.randrange(0,4572,50))

pretty_print_text_and_label(random.randrange(0,4572,100))

pretty_print_text_and_label(random.randrange(0,4572,200))

pretty_print_text_and_label(random.randrange(0,4572,500))

pretty_print_text_and_label(random.randrange(0,4572,800))

pretty_print_text_and_label(random.randrange(0,4572,1000))
Sincere_counts = Counter()

Insencere_counts = Counter()

Total_counts = Counter()

pp = pprint.PrettyPrinter(indent=4)
for i in range(len(df_data)):  #range(len(train)):

    if(df_data.iloc[i][2] == 0):

        for word in df_data.iloc[i][1].split(" "):

            Sincere_counts[word] += 1

            Total_counts[word] += 1

    else:

        for word in df_data.iloc[i][1].split(" "):

            Insencere_counts[word] += 1

            Total_counts[word] += 1
print("the most used word in insencere questions")

pp.pprint(Insencere_counts.most_common()[0:20])
print("the most used word in sencere questions")

pp.pprint(Sincere_counts.most_common()[0:20])
print("the most used word in all questions")

pp.pprint(Sincere_counts.most_common()[0:20])
vocab = set(Total_counts.keys())

vocab_size = len(vocab)

print(vocab_size)
vocab_vector = np.zeros((1, 200)) # np.zeros((1, vocab_size))

pp.pprint(vocab_vector.shape)

pp.pprint(vocab_vector)
word_column_dict = {}

p=0

for word,count in list(Total_counts.most_common()[0:200]):

    # {key: value} is {word: column}

    word_column_dict[word] = p

    p+=1
def update_input_layer(text):

    

    global vocab_vector

    

    # clear out previous state, reset the vector to be all 0s

    vocab_vector *= 0

    for word in text.split(" "):

        if word in word_column_dict:

            vocab_vector[0][word_column_dict[word]] += 1

        

    return vocab_vector.tolist()[0]



# example 

print(update_input_layer("the the"))
X_train = [update_input_layer(df_data.iloc[i][1]) for i in range(len(df_data))]
y_train = [df_data.iloc[i][2] for i in range(len(df_data))]
from keras import Sequential

from keras.layers import Dense

model = Sequential()  # a basic feed-forward model

#model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784



model.add(Dense(128,input_dim=200, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation

model.add(Dense(1, activation=tf.nn.sigmoid))  # our output layer. 10 units for 10 classes. Softmax for probability distribution



model.compile(optimizer='adam',  # Good default optimizer to start with

              loss='binary_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.

              metrics=['accuracy'])  # what to track
model.summary()
history = model.fit(np.asarray(X_train), np.asarray(y_train), epochs = 100)
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
df_test.head()
X_test = [update_input_layer(df_test.iloc[i][1]) for i in range(len(df_test))]
predictions = model.predict(np.asarray(X_test))
k=0

for i in range(len(df_test)):

    if(predictions[i][0]>0.5 and df_test.iloc[i,2]==1):

        k+=1

    elif(predictions[i][0]<0.5 and df_test.iloc[i,2]==0):

        k+=1

  

print(k/len(df_test)*100, "% of testing data are well predicted")  
print("type \t : \t question\n")

for d in range(5):

    r = random.randrange(0,len(df_test))

    s = "sincere question  :"

    q = "===> predicted ===> sincere question "

    if df_test.iloc[r,2]==1:

        s = "insincere question  :"

    if predictions[r][0]>0.5 : 

        q = "===> predicted   ===> insincere question "

        

    print(s+df_test.iloc[r,1]+q)
