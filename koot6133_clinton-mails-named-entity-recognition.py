# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

import itertools

import re

from operator import itemgetter



nlp = en_core_web_sm.load()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_mail = pd.read_csv('../input/hillary-clinton-emails/Emails.csv', header=0)

data_mail.head(5)
BodyText = data_mail.ExtractedBodyText

ex = BodyText[1398]

print(ex)

doc = nlp(ex)

doc
print([(X.text, X.label_) for X in doc.ents])  
displacy.render(nlp(ex), jupyter=True, style='ent')
displacy.render(nlp(ex), style='dep', jupyter = True, options = {'distance': 120})
def dicReplace(words, replacements, string):

    #string = ex

    #words = entityArray

    #replacements = mailMatch

    if words.size > 0:

        dic_NER = dict(zip(words, replacements))

        rep = dic_NER

        # use these three lines to do the replacement

        rep = dict((re.escape(k), v) for k, v in rep.items())

        pattern = re.compile("|".join(rep.keys()))

        string = pattern.sub(lambda m: rep[re.escape(m.group(0))], string)

        transformedString = pattern.sub(lambda m: rep[re.escape(m.group(0))], string)

    else: transformedString = string

    return(transformedString)
def NERTransformer(string):

    ent_list = [(X.text, X.label_) for X in nlp(ex).ents]  

    entityArray= np.array([item[1] for item in  ent_list]) 

    wordArray = np.array([item[0] for item in  ent_list])

    poIndex = [(entityArray == 'ORG') | (entityArray == 'PERSON')]

    entityArray2 = entityArray[tuple(poIndex)]

    wordArray2 = wordArray[tuple(poIndex)]

    transformedString= dicReplace(wordArray2, entityArray2, string)

    return(transformedString)
data_names= pd.read_csv('../input/englishnames3/englishNames.csv', header=0)

data_names['Name'] = ' ' + data_names['Name'].astype(str) + ' ' #add spaces 

data_names['Entity'] = ' ' + data_names['Entity'].astype(str) + ' ' #add spaces 

data_names
def NamesTransformer(string):

    transformedString = dicReplace(data_names['Name'], data_names['Entity'], string)

    return(transformedString)
def EmailTransformer(string):

    mailMatch = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', string)

    mailMatch = np.array(mailMatch)

    entityArray = np.repeat('EMAILADDRESS', len(mailMatch))

    transformedString = dicReplace(mailMatch,entityArray, string)

    return(transformedString)
def anonymous(string):

    stringAno1 = NERTransformer(string)

    stringAno2 = NamesTransformer(stringAno1)

    stringAno3 = EmailTransformer(stringAno2)

    return(stringAno3)
anonymous(ex)
#BodyText2 = BodyText[~BodyText.isnull()] #remove nans

#anonymous(BodyText2[2556])

#BodyText2.apply(anonymous)