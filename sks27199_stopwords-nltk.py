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
import nltk

from nltk.corpus import stopwords

from nltk.corpus import wordnet

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

input_sent = input("Enter:\n")
import re

def pre_process(text):

    

    # lowercase

    text=text.lower()

    

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    

    # remove special characters and digits

    text=re.sub("(\\d|\\W)+"," ",text)

    

    return text


input_sent = pre_process(input_sent)

  

stop_words = set(stopwords.words('english')) 

  

word_tokens = word_tokenize(input_sent) 

  

filtered_sentence = [w for w in word_tokens if not w in stop_words] 

  

print(word_tokens)

print(filtered_sentence) 
# Lemmatize with POS Tag



def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)





# 1. Init Lemmatizer

lemmatizer = WordNetLemmatizer()



# 2. Lemmatize a Sentence with the appropriate POS tag

keyset = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in filtered_sentence]

keyset = list(np.unique(np.array(keyset)))

keyset
keysetsyn=[]

for key in keyset:

    for syn in wordnet.synsets(key):

        for name in syn.lemma_names():

            keysetsyn.append(name)

unique_keyset = list(np.unique(np.array(keysetsyn)))

unique_keyset
keywords = filtered_sentence + unique_keyset

keywords = list(np.unique(np.array(keywords)))

keywords


