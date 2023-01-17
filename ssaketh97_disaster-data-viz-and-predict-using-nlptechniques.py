import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sb

import re

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.info()
top = train.groupby('keyword')['id'].count()

top = pd.DataFrame({'keyword':top.index,'count':top.values}).sort_values(by=['count']).tail(20)



bottom = train.groupby('keyword')['id'].count()

bottom = pd.DataFrame({'keyword':bottom.index,'count':bottom.values}).sort_values(by=['count']).head(20)



plt.figure(figsize=(12,10))



plt.subplot(211)

barlist = plt.bar(data=top, x = 'keyword',height = 'count',color = 'cadetblue')

plt.xticks(rotation = 20);

plt.ylabel('count')

plt.title('Top20 unique keywords')

barlist[0].set_color('darkgoldenrod');

barlist[2].set_color('indianred');

barlist[3].set_color('indianred');

barlist[15].set_color('darkgoldenrod');

barlist[9].set_color('darkslategrey');

barlist[18].set_color('darkseagreen');



plt.subplot(212)

barlist = plt.bar(data=bottom, x = 'keyword',height = 'count', color = 'cadetblue');

plt.xticks(rotation = 45);

plt.ylabel('count');

plt.title('Bottom20 unique keywords')

barlist[14].set_color('darkslategrey');

barlist[10].set_color('darkseagreen');



sb.despine(left = True, bottom  = True)

plt.tight_layout()



print(str(train['keyword'].nunique())+ ' total unique keywords')
top = train.groupby('location')['id'].count()

top = pd.DataFrame({'location':top.index,'count':top.values}).sort_values(by=['count']).tail(20)





plt.figure(figsize=(16,6))



barlist = plt.bar(data=top, x = 'location',height = 'count', color = 'cadetblue')

plt.xticks(rotation = 90);

plt.ylabel('count')

plt.title('Top20 unique locations')



barlist[1].set_color('darksalmon')

barlist[4].set_color('darksalmon')

barlist[3].set_color('peru')

barlist[18].set_color('peru')

barlist[17].set_color('dimgrey')

barlist[19].set_color('dimgrey')



sb.despine(left = True, bottom  = True)



print(str(train['location'].nunique())+ ' total unique locations')
train.loc[5,'text']
train.loc[31,'text']
train.loc[38,'text']
def find_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return (url.search(text) != None)



def clean_text(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    

    url = re.compile(r'https?://\S+|www\.\S+')

    text = url.sub(r'',text)

    

    text = text.replace('#',' ')

    text = text.replace('@',' ')

    symbols = re.compile(r'[^A-Za-z0-9 ]')

    text = symbols.sub(r'',text)

    

    return text



def lemma(text):

    txt1 = wordnet_lemmatizer.lemmatize(text,pos=wordnet.NOUN)

    txt2 = wordnet_lemmatizer.lemmatize(text,pos=wordnet.VERB)

    txt3 = wordnet_lemmatizer.lemmatize(text,pos=wordnet.ADJ)

    if(len(txt1) < len(txt2) and len(txt1) < len(txt3)): 

        text = txt1

    elif(len(txt2) < len(txt1) and len(txt2) < len(txt3)):

        text = txt2

    elif(len(txt3) < len(txt1) and len(txt3) < len(txt2)):

        text = txt3

    else:

        text = txt1   

    

    return text
#1. Replace missing values

train['keyword'].fillna('None', inplace=True)

train['location'].fillna('None', inplace=True)



#3. Replace %20 in the keyword column

train['keyword'] = train['keyword'].str.replace('%20','')



#4. location column handling

for ind in range(train.shape[0]):

    train.loc[ind,'location'] = train.loc[ind,'location'].split(',')[0]



#5,6,7. Text column handling

for ind in range(train.shape[0]):

    train.loc[ind,'tags_count'] = len(train.loc[ind,'text']) -  len(train.loc[ind,'text'].replace('#',''))

    train.loc[ind,'@_count'] = len(train.loc[ind,'text']) -  len(train.loc[ind,'text'].replace('@',''))

    train.loc[ind,'http_link'] =  find_URL(train.loc[ind,'text'])

    

train['text'] = train['text'].apply(lambda x: clean_text(x))

train.head(70)



#2 lemmatize keyword

train['keyword'] = train['keyword'].apply(lambda x: lemma(x))





train.head(10)