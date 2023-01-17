

import pandas as pd

import numpy as np

import re

import string

import matplotlib

%matplotlib inline  

matplotlib.get_backend()

metadata = pd.read_csv("../input/gender-classifier-DFE-791531.csv",encoding='latin1')
metadata.head(2)
metadata.columns

data = pd.read_csv("../input/gender-classifier-DFE-791531.csv",usecols= [0,5,19,17,21,10,11],encoding='latin1')
data.head(2)
def cleaning(s):

    s = str(s)

    s = s.lower()

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W,\s',' ',s)

    s = re.sub(r'[^\w]', ' ', s)

    s = re.sub("\d+", "", s)

    s = re.sub('\s+',' ',s)

    s = re.sub('[!@#$_]', '', s)

    s = s.replace("co","")

    s = s.replace("https","")

    s = s.replace(",","")

    s = s.replace("[\w*"," ")

    return s



data['Tweets'] = [cleaning(s) for s in data['text']]

data['Description'] = [cleaning(s) for s in data['description']]



from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

data['Tweets'] = data['Tweets'].str.lower().str.split()

data['Tweets'] = data['Tweets'].apply(lambda x : [item for item in x if item not in stop])

data.head(2)
data.gender.value_counts()
Male = data[data['gender'] == 'male']

Female = data[data['gender'] == 'female']

Brand = data[data['gender'] == 'brand']

Male_Words = pd.Series(' '.join(Male['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]

Female_Words = pd.Series(' '.join(Female['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]

Brand_words = pd.Series(' '.join(Brand['Tweets'].astype(str)).lower().split(" ")).value_counts()[:10]



Female_Words





Female_Words.plot(kind='bar',stacked=True, colormap='OrRd')
Male_Words
Male_Words.plot(kind='bar',stacked=True, colormap='plasma')
Brand_words
Brand_words.plot(kind='bar',stacked=True, colormap='Paired')