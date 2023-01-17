# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import spacy

import pandas as pd

import numpy as np



nlp = spacy.load('en')
!python -m spacy download en_core_web_md
df = pd.read_csv('../input/newyork-room-rentalads/room-rental-ads.csv')

df.head()
def tokenise(msg):

    doc = nlp(msg)

    

    for token in doc:

        print(token)
tokenise("I like planes and tanks")
from spacy.lang.en.stop_words import STOP_WORDS



print(STOP_WORDS)
def stop_remove(msg):

    doc = nlp(msg)

    

    for token in doc:

        if token.is_stop:

            pass

        else:

            print(token)
stop_remove("I have liked reading books for a good few years now")
def normalize(msg):

    

    doc = nlp(msg)

    res=[]

    

    for token in doc:

        if(token.is_stop or token.is_digit or token.is_punct or not(token.is_oov)):

            pass

        else:

            res.append(token.lemma_.lower()) #Lower case of the token lemma is added

    

    return " ".join(res)
import re



def normalize_1(msg):

    

    msg = re.sub('[^A-Za-z]+', ' ', str(msg)) #remove special character and intergers

    doc = nlp(msg)

    res=[]

    for token in doc:

        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 2): #Remove Stopwords, Punctuations, Currency and Spaces

            pass

        else:

            res.append(token.lemma_.lower())

    return res
df['Description'] = df['Description'].apply(normalize_1)

df.head()
from spacy import displacy



def displacify(msg):

    

    doc = nlp(msg)

    

    return displacy.render(doc, style='ent')
displacify("spaCy is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython. The library is published under the MIT license and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion.")