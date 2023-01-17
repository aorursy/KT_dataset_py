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

import glob  # to get a list of current files

import numpy as np  # nump package
def combiner(folder):

    

    read_files = glob.glob(folder + r"/*.txt") # reading all the files into one list

    # clean later the following line     

    #  the following piece of code has two functions

    #  appends end of line word to each sentence

    #  get a list of length to be used for choosing window size for LSTM

    cache = {}  # a dictionary developed to keep the analytics

    corpus_list = []  # corpus in a string 

    num_words = [] # empty array to save number of words

    num_chars = [] # empty array to save number of characters

    for f in read_files:

        with open(f, "r", encoding="utf8") as infile:

            for sentence in infile:

                if not sentence.strip(): continue  # skipping over empt lines

                sentence = sentence.strip()

                corpus_list.append(sentence)  # adding the current sentence to the corpus

                list_of_words = sentence.split(" ")  # list of words in sentence

                num_words.append(len(list_of_words))  # number of words in list

                num_chars.append(len(sentence))  # umber of characters in sentence

    # we first delete lines with small number of words

    corpus_list = [corpus_list[i] for i, n in enumerate(num_words) if n > 1]          

    corpus = " . ".join(corpus_list)            

    #  analytis on the poem length

    mean_length = np.mean(num_chars)  # mean of number of words   

    median_length = np.median(num_chars)  # median of the number of words

    #  developing the cache

    cache['median'] = median_length  # median of the sentence

    cache['mean'] = mean_length  # mean of the sentence

    cache['max_length'] = np.max(num_chars)

    

    return corpus, cache,  median_length
combiner('input')