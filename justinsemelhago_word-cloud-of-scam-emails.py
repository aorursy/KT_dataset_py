import matplotlib.pyplot as plt

import wordcloud

import logging

import collections

import re

import nltk

import pandas as pd

import numpy as np

from PIL import Image





logger = logging.getLogger('FraudEmails')



#Minimum occurrence of a word to appear in the WordCloud

frequency = 100

#Number of words in WordCloud

numb_of_words = 400
#Open and convert file to string (ignore strange characters)

try:

    with open('../input/fraudulent-email-corpus/fradulent_emails.txt','r',encoding='utf-8',errors='ignore') as file:

        text = file.read()

except Exception as e:

    logger.error('Process failed with error: '+repr(e))

finally:

    file.close()



#Delete everything between From r [.*?] Status: ?O

emails = re.sub('From.*?Status: ?O','',text,flags=re.DOTALL)
#Convert to lower case

emails = emails.lower()
#Tokenize into string and remove stop words (the, a, on...)

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

tokens = tokenizer.tokenize(emails)

emailWordList = list(filter(lambda word: word not in nltk.corpus.stopwords.words('english'), tokens))
#Remove anything less than two characters

emailWordList = list(filter(lambda word: len(word)>2, emailWordList))
#HTML tags modified

htmldf = pd.read_csv('../input/html-tags/html_tags.txt', sep='\t').dropna()

htmldf['Tag'] = htmldf['Tag'].str.strip()

htmldf.head()
#Remove HTML tags

finalEmailFormat = [word for word in emailWordList if '<'+word.strip()+'>' not in htmldf['Tag'].to_list()]
#Correspond every entry to natural language tag

tags = nltk.pos_tag(finalEmailFormat)
#Every word dataframe

df = pd.DataFrame(tags, columns=['word','tag'])

df.head()
#Verb dataframe

dfverb = df[df['tag']=='VB']

dfverb = dfverb.word.value_counts().reset_index()

dfverb = dfverb[dfverb['word']>frequency]

dfverb.head()
#Noun dataframe

dfnoun = df[df['tag']=='NN']

dfnoun = dfnoun.word.value_counts().reset_index()

dfnoun = dfnoun[dfnoun['word']>frequency]

dfnoun.head()
#Adjective dataframe

dfadj = df[df['tag']=='JJ']

dfadj = dfadj.word.value_counts().reset_index()

dfadj = dfadj[dfadj['word']>frequency]

dfadj.head()
#Noun plural dataframe

dfnounpl = df[df['tag']=='NNS']

dfnounpl = dfnounpl.word.value_counts().reset_index()

dfnounpl = dfnounpl[dfnounpl['word']>frequency]

dfnounpl.head()
#Keep VB (verb), NN (singular nouns), JJ (adjectives), NNS (plural nouns)

toKeep = ['VB','NN','JJ','NNS']



#Get only tag if tag is associated with above list

forWordCloud = [tag[0] for tag in tags if tag[1] in toKeep]

#Create dictionary with count of each entry

scamDict = collections.Counter(forWordCloud)

#Remove website words

toRemove = ['nbsp','http','charset','iso','html','www']



for removal in toRemove:

    del scamDict[removal]
#Remove less than 'frequency' instances

scamDict = {k: v for k, v in scamDict.items() if v > frequency}

#Open mask and create word cloud

africaMask = np.array(Image.open('../input/africaoutline/Africa-outline.jpg'))

scamCloud = wordcloud.WordCloud(width=1600,height=1600,

                                max_words=numb_of_words,mask=africaMask,

                                contour_width=1,contour_color='blue').generate_from_frequencies(scamDict)

#Create image

plt.figure(figsize=(30,30))

plt.imshow(scamCloud)

plt.axis('off')

plt.savefig('./ScamWordCloud.png',bbox_inches='tight',pad_inches=0,dpi=133)

plt.show()
#Output formatted word frequency

outputformatdf = pd.DataFrame.from_dict(scamDict, orient='index').sort_values(by=[0],ascending=False)

outputformatdf.to_csv('./WordFrequency_Processed.csv',header=False)

outputformatdf.head()