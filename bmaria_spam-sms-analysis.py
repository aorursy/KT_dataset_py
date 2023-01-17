# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import string

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.classify import NaiveBayesClassifier

from nltk.classify.util import accuracy

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#read from csv

df=pd.read_csv("../input/spam.csv",encoding ='latin-1')



# change the target from string to int

df['v1'] = df['v1'].astype('category')

df['v1'] = pd.Categorical(df['v1']).codes



#convert to string object

df['v2']=df['v2'].astype(str)

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, axis=1)



#punctuation counts

#df['punc_counts'] = df['v2'].apply(lambda x:len([c for c in x if c in string.punctuation]))
#stop words removal

english_stops = set(stopwords.words('english'))

df['words']=df['v2'].apply(lambda text: dict([(word, True) for word in text.split() if word not in english_stops]))
#create spam features dictionary

spam_feats=[(row.words,'spam') for idx, row in df[df.v1==1].iterrows()]

#create ham features dictionary

ham_feats=[(row.words,'ham') for idx, row in df[df.v1==0].iterrows()]



#combined the features set

train_feats = spam_feats + ham_feats

#create NB classifier instance

classifier = NaiveBayesClassifier.train(train_feats)
#spam sms test

print(classifier.classify({'entitled':True, 'win':True, '3750':True, 'pounds':True}))



#ham sms test

print(classifier.classify({'this':True, 'is':True, 'test':True, 'message':True}))
#word cloud plot

from wordcloud import WordCloud



spam_words = ''



for text in df[df.v1==1]['v2']:

    tokens = word_tokenize(text)

    for words in tokens:

        spam_words = spam_words + words + ' '



wc = WordCloud(background_color="white", max_words=200, max_font_size=40,random_state=1, collocations=False)

wc.generate(spam_words)

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.figure()

plt.show()