!pip install PySastrawi
import numpy as np

import pandas as pd



!ls '../input'
data = pd.read_csv('../input/indonesian-abusive-and-hate-speech-twitter-text/data.csv', encoding='latin-1')



alay_dict = pd.read_csv('../input/indonesian-abusive-and-hate-speech-twitter-text/new_kamusalay.csv', encoding='latin-1', header=None)

alay_dict = alay_dict.rename(columns={0: 'original', 

                                      1: 'replacement'})



id_stopword_dict = pd.read_csv('../input/indonesian-stoplist/stopwordbahasa.csv', header=None)

id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})
print("Shape: ", data.shape)

data.head(15)
data.HS.value_counts()
data.Abusive.value_counts()
print("Toxic shape: ", data[(data['HS'] == 1) | (data['Abusive'] == 1)].shape)

print("Non-toxic shape: ", data[(data['HS'] == 0) & (data['Abusive'] == 0)].shape)
print("Shape: ", alay_dict.shape)

alay_dict.head(15)
print("Shape: ", id_stopword_dict.shape)

id_stopword_dict.head()

import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()

stemmer = factory.create_stemmer()



def lowercase(text):

    return text.lower()



def remove_unnecessary_char(text):

    text = re.sub('\n',' ',text) # Remove every '\n'

    text = re.sub('rt',' ',text) # Remove every retweet symbol

    text = re.sub('user',' ',text) # Remove every username

    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL

    text = re.sub('  +', ' ', text) # Remove extra spaces

    return text

    

def remove_nonaplhanumeric(text):

    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 

    return text



alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))

def normalize_alay(text):

    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])



def remove_stopword(text):

    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])

    text = re.sub('  +', ' ', text) # Remove extra spaces

    text = text.strip()

    return text



def stemming(text):

    return stemmer.stem(text)



print("remove_nonaplhanumeric: ", remove_nonaplhanumeric("Halooo,,,,, duniaa!!"))

print("lowercase: ", lowercase("Halooo, duniaa!"))

print("stemming: ", stemming("Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan"))

print("remove_unnecessary_char: ", remove_unnecessary_char("Hehe\n\n RT USER USER apa kabs www.google.com\n  hehe"))

print("normalize_alay: ", normalize_alay("aamiin adek abis"))

print("remove_stopword: ", remove_stopword("ada hehe adalah huhu yang hehe"))
def preprocess(text):

    text = lowercase(text) # 1

    text = remove_nonaplhanumeric(text) # 2

    text = remove_unnecessary_char(text) # 2

    text = normalize_alay(text) # 3

    text = stemming(text) # 4

    text = remove_stopword(text) # 5

    return text
data['Tweet'] = data['Tweet'].apply(preprocess)
print("Shape: ", data.shape)

data.head(15)
data.to_csv('preprocessed_indonesian_toxic_tweet.csv', index=False)