import time

start = time.clock()



#open data

import pandas as pd

import numpy as np

import nltk

import codecs

from nltk.stem.snowball import SnowballStemmer

print(SnowballStemmer("english").stem("generously"))

from nltk.tokenize import word_tokenize



datas = pd.read_csv('../input/train.csv') #

datas = datas.fillna('leeg')

datas=datas[:100000]

def cleantxt(x):    # aangeven sentence

    # Removing non ASCII chars

    x = x.replace(r'[^\x00-\x7f]',r' ') 

#    x = x.decode('utf-8').strip()

    x = x.lower()

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        x = x.replace(char, ' ' + char + ' ')

    return x



datas['question1']=datas['question1'].map(cleantxt)

datas['question2']=datas['question2'].map(cleantxt)

print(datas)



end = time.clock()

print('open:',end-start)
from nltk.corpus import stopwords # Import the stop word list

#print stopwords.words("english") 



for xyz in range(len(datas)):

    q1=datas.iloc[xyz].question1

    q2=datas.iloc[xyz].question2

    sent1 = nltk.wordpunct_tokenize(q1) #tokenize sentence

    sent2 = nltk.wordpunct_tokenize(q2) 

    sims=pd.DataFrame(data=None, index=sent1, columns=sent2)  #abs(np.random.randn(len(sent1), len(sent2))/1000000)

    equq1 = [w for w in sent1 if w in sent2]

    difq1 = [w for w in sent1 if w not in sent2]

    difq2 = [w for w in sent2 if w not in sent1]

    diftot = difq1+difq2

    difton = [w for w in diftot if not w in stopwords.words("english")]

    Q2no = [w for w in sent2 if not w in stopwords.words("english")]

    if len(difton)==0 and datas.iloc[xyz].is_duplicate==0:

        print('overlap irrelevant ?',q1,q2,datas.iloc[xyz].is_duplicate)

    if len(Q2no)<3 :

        print('short question ?',q1,q2,datas.iloc[xyz].is_duplicate)

    