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
trump = open('/kaggle/input/Trump_Austin_Aug-23-16.txt', encoding='utf8').read()

#display the data

 

print(trump)

 
corpus = trump.split()

 

#Display the corpus

print(corpus)
def make_pairs(corpus):

    for i in range(len(corpus) - 1):

        yield (corpus[i], corpus[i + 1])

pairs = make_pairs(corpus)
print(pairs)

word_dict = {}

for word_1, word_2 in pairs:

    if word_1 in word_dict.keys():

        word_dict[word_1].append(word_2)

    else:

        word_dict[word_1] = [word_2]
#randomly pick the first word

first_word = np.random.choice(corpus)

 

#Pick the first word as a capitalized word so that the picked word is not taken from in between a sentence

while first_word.islower():

    first_word = np.random.choice(corpus)

#Start the chain from the picked word

chain = [first_word]

#Initialize the number of stimulated words

n_words = 20

print(chain)
for i in range(n_words): 

    chain.append(np.random.choice(word_dict[chain[-1]]))
#Join returns the chain as a string

print(' '.join(chain))