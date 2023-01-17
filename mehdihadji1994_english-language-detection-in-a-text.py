!pip install spacy_cld
import numpy as np 

import pandas as pd

import spacy

from spacy_cld import LanguageDetector





import langdetect

import langid



import os

print(os.listdir("../input"))
devoxxfr = pd.read_csv('../input/keyword-DevoxxFR.csv')

devoxxfr.head()
tweets    = devoxxfr['tweets']
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



tweets          = devoxxfr['tweets']

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
devoxxfr['languages_spacy'] = languages_spacy

devoxxfr['languages_langdetect'] = languages_langdetect
devoxxfr.head()
devoxxfr['languages_spacy'].value_counts()
devoxxfr['languages_langdetect'].value_counts()
devoxxfr.to_csv('devoxxfr-languages.csv',index=False)