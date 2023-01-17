#@title Import our libraries (this may take a minute or two)

import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv). 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import string

import nltk

import spacy

import wordcloud

import os # Good for navigating your computer's files 

import sys



from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from spacy.lang.en.stop_words import STOP_WORDS

nltk.download('wordnet')

nltk.download('punkt')



from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

!python -m spacy download en_core_web_md

import en_core_web_md
data_file  = '../input/yelp-reviews-dataset/yelp.csv'

# read our data in using 'pd.read_csv('file')'

yelp = pd.read_csv(data_file)
yelp.head()
#Remove unnecessary columns

yelp.drop(labels=['business_id','date','review_id','type','user_id'],inplace=True,axis=1)
num_stars = 1 #@param {type:"integer"}



for t in yelp[yelp['stars'] == num_stars]['text'].head(20).values:

    print (t) 
rule_1 = "Bad" 

rule_2 = "Good" 

rule_3 = "Fine" 
#Word cloud for differently rated reviews

num_stars =  2 

this_star_text = ''

for t in yelp[yelp['stars'] == num_stars]['text'].values: # form field cell

    this_star_text += t + ' '

    

wordcloud = WordCloud()    

wordcloud.generate_from_text(this_star_text)

plt.figure(figsize=(14,7))

plt.imshow(wordcloud, interpolation='bilinear')
example_text = "All the people I spoke to were super nice and very welcoming." #@param {type:"string"}

tokens = word_tokenize(example_text)

tokens
example_word = "ok"

if example_word.lower() in STOP_WORDS:

  print (example_word + " is a stop word.")

else:

  print (example_word + " is NOT a stop word.")
nlp = en_core_web_md.load()

doc = nlp(u"We are running out of time! Are we though?")

doc
doc = nlp(u"We are running out of time! Are we though?")

token = doc[0] # Get the first word in the text.

assert token.text == u"We" # Check that the token text is 'We'.

assert len(token) == 2 # Check that the length of the token is 2.
doc = nlp(u"I like apples")

apples = doc[2]



print(apples.vector.shape[0]) # Each word is being represented by 96 dimensional vector embedding
doc = nlp(u'dog and cat')

word1 = doc[0]

word2 = doc[2]

word1.similarity(word2)
yelp = yelp[yelp.stars != 4]
def is_good_review(stars):

    if stars == 5:  ### TODO: FILL IN THE IF STATEMENT HERE ###:

        return True

    else:

        return False



# Change the stars field to either be 'good' or 'bad'.

yelp['is_good_review'] = yelp['stars'].apply(is_good_review)
#@title Run this to see the one-hot encoding of 'great tacos at this restaurant'

print('{:^5}|{:^5}|{:^4}|{:^4}|{:^10}'.format('great', 'tacos', 'at','this','restaurant'))

print('--------------------------------------------')

print('{:^5}|{:^5}|{:^4}|{:^4}|{:^10}'.format('1', '0', '0','0','0'))

print('{:^5}|{:^5}|{:^4}|{:^4}|{:^10}'.format('0', '1', '0','0','0'))

print('{:^5}|{:^5}|{:^4}|{:^4}|{:^10}'.format('0', '0', '1','0','0'))

print('{:^5}|{:^5}|{:^4}|{:^4}|{:^10}'.format('0', '0', '0','1','0'))

print('{:^5}|{:^5}|{:^4}|{:^4}|{:^10}'.format('0', '0', '0','0','1'))