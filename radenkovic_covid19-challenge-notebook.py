# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

filelist = []



# LOAD .json file paths into array

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        extension = os.path.splitext(filename)[1];

        if extension == '.json':

            filelist.append(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.

print('Total JSON files', len(filelist))
import json



# Create DataFrame -- TODO add more cols if needed

df = pd.DataFrame(columns=[ "paper_id", "text"])





# FILTER if includes string in the body text

filter_by_string = 'vaccine'



# NOTE: trying on 100 first files only

# TODO: bump this number when ready

for file in filelist[:100]:

    f = open(file)

    raw = f.read()

    loaded_json = json.loads(raw)



    # Create body text (joined)

    collector = ''

    for item in loaded_json['body_text']:

        collector = collector + ' ' +item['text']



    # Try to filter

    if (filter_by_string in collector):

        entry = {

        'paper_id': loaded_json['paper_id'],

        'text': collector

        }

        df = df.append(entry, ignore_index=True)    





print ('Total files including word', filter_by_string, len(df))

df.sample(3)

import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize, sent_tokenize





freqTable = dict()

def create_frequency_table(text_string):

    stopWords = set(stopwords.words("english"))

    words = word_tokenize(text_string)

    ps = PorterStemmer()

    for word in words:

        word = ps.stem(word)

        if word in stopWords:

            continue

        if word in freqTable:

            freqTable[word] += 1

        else:

            freqTable[word] = 1



## Iterate and create Frequency table

for txt in df['text']:

    create_frequency_table(txt)

            



freq = pd.DataFrame.from_dict(freqTable, orient='index')

freq.reset_index(level=0, inplace=True)

freq.columns = ['word', 'frequency']



freq['word_length'] = freq['word'].apply(lambda x: len(x))



# Sort & Remove short ones

freq = freq.where(freq['word_length'] > 5).sort_values('frequency', ascending=0).reset_index()

freq = freq.drop('index', axis='columns')

freq = freq[0:30] # we care about top 30

freq.head(10)
import spacy



nlp = spacy.load("en_core_web_sm")





freq_table2 = dict()

for txt in df['text']:

    doc = nlp(txt)

    nouns = [chunk.text for chunk in doc.noun_chunks]

    for word in nouns:

        if word in freq_table2:

            freq_table2[word] += 1

        else:

            freq_table2[word] = 1



# for token in doc:

#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,

#             token.shape_, token.is_alpha, token.is_stop)



freq = pd.DataFrame.from_dict(freq_table2, orient='index')

freq.reset_index(level=0, inplace=True)

freq.columns = ['word', 'frequency']





# Sort & Remove short ones

freq = freq.sort_values('frequency', ascending=0).reset_index()

freq = freq.drop('index', axis='columns')

freq = freq[0:30] # we care about top 30

freq.head(10)