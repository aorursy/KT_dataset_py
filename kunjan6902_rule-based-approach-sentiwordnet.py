import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd



fn = r"/kaggle/input/twitter-product-sentiment-analysis/Twitter Product Sentiment Analysis.csv"

df = pd.read_csv(fn)



print(df.shape)

print(df.head(3))
import nltk

from nltk.corpus import wordnet as wn

from nltk.corpus import sentiwordnet as swn

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



nltk.download('wordnet')

nltk.download('sentiwordnet')

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
!pip install tweet-preprocessor

import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.SMILEY, p.OPT.EMOJI)
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

cleantweets = []

postags = []

import string

table = str.maketrans('','', string.punctuation)



for tweet in df['tweet']:

    try:

        tweet = p.clean(tweet)

        #tokenize + lower case

        words2 = word_tokenize(tweet.lower())

        #remove puncts

        words3 = [w.translate(table) for w in words2]

        #remove stopwords

        words4 = [word for word in words3 if word not in stop_words]

        #applying lemmatization

        words5 = [lemmatizer.lemmatize(word) for word in words4]

                

        #combining all words

        cleantweets.append((" ".join(words5)).strip())

    except:

        cleantweets.append(tweet)

        continue



print(len(cleantweets))



df['clean_Tweets'] = cleantweets
pos=neg=obj=count=0



postagging = []



for tweet in df['clean_Tweets']:

    list = word_tokenize(tweet)

    postagging.append(nltk.pos_tag(list))



df['pos_tags'] = postagging
# Convert between the PennTreebank tags to simple Wordnet tags

def penn_to_wn(tag):

    if tag.startswith('J'):

        return wn.ADJ

    elif tag.startswith('N'):

        return wn.NOUN

    elif tag.startswith('R'):

        return wn.ADV

    elif tag.startswith('V'):

        return wn.VERB

    return None
# Returns list of pos-neg and objective score. But returns empty list if not present in senti wordnet.

def get_sentiment(word,tag):

    wn_tag = penn_to_wn(tag)

    

    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):

        return []



    #Lemmatization

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)

    if not lemma:

        return []



    #Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet. 

    #Synset instances are the groupings of synonymous words that express the same concept. 

    #Some of the words have only one Synset and some have several.

    synsets = wn.synsets(word, pos=wn_tag)

    if not synsets:

        return []



    # Take the first sense, the most common

    synset = synsets[0]

    swn_synset = swn.senti_synset(synset.name())



    return [synset.name(), swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]
pos=neg=obj=count=0

senti_score = []



for pos_val in df['pos_tags']:

    senti_val = [get_sentiment(x,y) for (x,y) in pos_val]

    for score in senti_val:

        try:

            pos = pos + score[1]  #positive score is stored at 2nd position

            neg = neg + score[2]  #negative score is stored at 3rd position

        except:

            continue

    senti_score.append(pos - neg)

    pos=neg=0    

    

df['senti_score'] = senti_score
df.head
df.to_csv("Sentiment_predited_result.csv")