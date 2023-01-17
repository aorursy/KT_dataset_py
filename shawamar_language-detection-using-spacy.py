!pip install spacy_cld
import numpy as np 

import pandas as pd

import re

import spacy

from spacy_cld import LanguageDetector





import langdetect

import langid



import os

print(os.listdir("../input"))
# function for data cleaning..

def remove_xml(text):

    return re.sub(r'<[^<]+?>', '', text)



def remove_newlines(text):

    return text.replace('\n', ' ') 

    



def remove_manyspaces(text):

    return re.sub(r'\s+', ' ', text)



def clean_text(text):

    text = remove_xml(text)

    text = remove_newlines(text)

    text = remove_manyspaces(text)

    return text
df = pd.read_csv('../input/data.csv')

df.head()
#Loading dataset  in tweets.

tweets    = df['tweets']
for line in tweets:

    line=clean_text(line)

df.head()
"""

result = str(result[0])[:2] : keeping the most dominant language wich is situated 

in the 1st index, and we store the first 2 characters

"""



languages_langdetect = []



# the try except blook because there is some tweets contain links

for line in tweets:

    try:

        result = langdetect.detect_langs(line)

        result = str(result[0])[:2]

    except:

        result = 'unknown'

    

    finally:

        languages_langdetect.append(result)
nlp = spacy.load('en')

language_detector = LanguageDetector()

nlp.add_pipe(language_detector)
"""

doc._.languages returns : list of str

like : ['fr'] -> french

       ['en'] -> english

       [] -> empty

       ['fr','en'] -> french (the most dominant in a tweet) and english (least dominant)

"""



tweets          = df['tweets']

languages_spacy = []



for e in tweets:

    doc = nlp(e)

    # cheking if the doc._.languages is not empty

    # then appending the first detected language in a list

    if(doc._.languages):

        languages_spacy.append(doc._.languages[0])

    # if it is empty, we append the list by unknown

    else:

        languages_spacy.append('unknown')
df['languages_spacy'] = languages_spacy

df['languages_langdetect'] = languages_langdetect
df.head()
df['languages_spacy'].value_counts()
df['languages_langdetect'].value_counts()
df.to_csv('Detected_Languages.csv',index=False)