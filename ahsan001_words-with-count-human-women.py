import pandas as pd

import re

import nltk

from nltk.corpus import wordnet

from csv import QUOTE_NONE
def read_and_reformat(csv_path):

    df = pd.read_csv(csv_path,

                     sep='|',

                     encoding='iso-8859-1',

                     dtype=object,

                     header=None,

                     quoting=QUOTE_NONE,

                     names=['Surah', 'Ayah', 'Text'])    

    df['Text'] = df['Text'].str.replace('#NAME\?', '')

    df['Text'] = df['Text'].str.strip(',')

    return df
df = read_and_reformat('../input/English.csv')

df.head()
searchKeywords  = ['Human', 'Women', 'Humility','Heaven' , 'Hell']

for keyword in searchKeywords:     

    synonyms = []

    countTotal = 0

    for syn in wordnet.synsets(keyword):

        for l in syn.lemmas():

            countTotal += df['Text'].str.count(l.name() , re.IGNORECASE).sum() 

            synonyms.append(l.name())

    print(keyword + " = " + str(countTotal)   )

    print(synonyms)    
searchKeywords  = ['Human', 'Women', 'Humility','Heaven' , 'Hell']

for key in searchKeywords:        

    df[key] = df['Text'].str.count(key , re.IGNORECASE)

    print (key + " = " + str(df[key].sum()))