# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

nlp = spacy.load('en')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

print("Libraries Loaded")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
doc = nlp("Tea is healthy and calming, don't you think?")



for token in doc:

    print(token)
print(f"Token \tLemma \tStopword".format('Token','Lemma','Stopword'))

print("-"*26)

for token in doc:

    print(f"{str(token)}\t{token.lemma_}\t{token.is_stop}")

from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')



terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']

patterns =[nlp(text) for text in terms]

matcher.add("TerminologyList", None , *patterns)



text_doc = nlp("Glowing review overall, and some really interesting side-by-side "

               "photography tests pitting the iPhone 11 Pro against the "

               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 



matches = matcher(text_doc)

print(matches)



match_id, start, end = matches[0]

print(nlp.vocab.strings[match_id], text_doc[start:end])