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
vietnamese_file = '../input/vietnamese-stopwords/vietnamese-stopwords.txt'

with open(vietnamese_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

def plotWordFrequency(input):

    f = open(vietnamese_file,'r')

    words = [x for y in [l.split() for l in f.readlines()] for x in y]

    data = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)[:40] 

    most_words = [x[0] for x in data]

    times_used = [int(x[1]) for x in data]

    plt.figure(figsize=(20,10))

    plt.bar(x=sorted(most_words), height=times_used, color = 'grey', edgecolor = 'black',  width=.5)

    plt.xticks(rotation=45, fontsize=18)

    plt.yticks(rotation=0, fontsize=18)

    plt.xlabel('Most Common Words:', fontsize=18)

    plt.ylabel('Number of Occurences:', fontsize=18)

    plt.title('Most Commonly Used Words: %s' % (vietnamese_file), fontsize=24)

    plt.show()
vietnamese_file = '../input/vietnamese-stopwords/vietnamese-stopwords.txt'

plotWordFrequency(vietnamese_file)
import pronouncing

import markovify

import re

import random

import numpy as np

import os

import keras

from keras.models import Sequential

from keras.layers import LSTM 

from keras.layers.core import Dense
def create_network(depth):

    model = Sequential()

    model.add(LSTM(4, input_shape=(2, 2), return_sequences=True))

    for i in range(depth):

        model.add(LSTM(8, return_sequences=True))

    model.add(LSTM(2, return_sequences=True))

    model.summary()

    model.compile(optimizer='rmsprop',

              loss='mse')

    if artist + ".rap" in os.listdir(".") and train_mode == False:

        model.load_weights(str(artist + ".rap"))

        print("loading saved network: " + str(artist) + ".rap") 

    return model
def markov(text_file):

    ######

    read = open(text_file, "r", encoding='utf-8').read()

    text_model = markovify.NewlineText(read)

    return text_model
def syllables(line):

    count = 0

    for word in line.split(" "):

        vowels = 'aeiouy'

#       word = word.lower().strip("!@#$%^&*()_+-={}[];:,.<>/?")

        word = word.lower().strip(".:;?!")

        if word[0] in vowels:

            count +=1

        for index in range(1,len(word)):

            if word[index] in vowels and word[index-1] not in vowels:

                count +=1

        if word.endswith('e'):

            count -= 1

        if word.endswith('le'):

            count+=1

        if count == 0:

            count +=1

    return count / maxsyllables