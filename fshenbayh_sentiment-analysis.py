#THIS IS AN INSERTED EDIT, JL



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from textblob import TextBlob



text = '''

The titular threat of The Blob has always struck me as the ultimate movie

monster: an insatiably hungry, amoeba-like mass able to penetrate

virtually any safeguard, capable of--as a doomed doctor chillingly

describes it--"assimilating flesh on contact.

Snide comparisons to gelatin be damned, it's a concept with the most

devastating of potential consequences, not unlike the grey goo scenario

proposed by technological theorists fearful of

artificial intelligence run rampant.

'''



blob = TextBlob(text) # create a textblob object and name it 'blob'
blob[0:50] # we can slice them
blob.upper(), blob.lower() # we can convert them to upper or lower case
blob.find("doomed") # we can locate particular terms
apple_blob = TextBlob('apples')  

apple_blob == 'apples' # we can make direct comparisons between TextBlobs and strings
blob.ngrams(n=2)
zen = TextBlob("Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex.")

zen.words, zen.sentences
blob.tags #returns tuple of (word, part-of-speech)
blob.noun_phrases
b = TextBlob("I havv verry goood speling!")

print(b.correct())
from textblob import Word

w = Word('conandrum')

w.spellcheck()
blob.sentiment
blob.sentiment.polarity
def get_textBlob_score(text):

    # Polarity score is between -1 to 1

    polarity_scores = []

    sents = TextBlob(text).sentences

    for sent in sents:

        polarity = sent.sentiment.polarity

        polarity_scores.append(polarity)

    return polarity_scores



text = '''

To be, or not to be, that is the question:

Whether 'tis nobler in the mind to suffer

The slings and arrows of outrageous fortune,

Or to take arms against a sea of troubles

And by opposing end them. To die—to sleep,

No more; and by a sleep to say we end

The heart-ache and the thousand natural shocks

That flesh is heir to: 'tis a consummation

Devoutly to be wish'd. To die, to sleep;

To sleep, perchance to dream—ay, there's the rub:

For in that sleep of death what dreams may come,

When we have shuffled off this mortal coil,

Must give us pause—there's the respect

That makes calamity of so long life.

'''



get_textBlob_score(text)
train = [

    ('I love this sandwich.', 'pos'),

    ('this is an amazing place!', 'pos'),

    ('I feel very good about these beers.', 'pos'),

    ('this is my best work.', 'pos'),

    ("what an awesome view", 'pos'),

    ('I do not like this restaurant', 'neg'),

    ('I am tired of this stuff.', 'neg'),

    ("I can't deal with this", 'neg'),

    ('he is my sworn enemy!', 'neg'),

    ('my boss is horrible.', 'neg')

 ]



test = [

    ('the beer was good.', 'pos'),

    ('I do not enjoy my job', 'neg'),

    ("I ain't feeling dandy today.", 'neg'),

    ("I feel amazing!", 'pos'),

    ('Gary is a friend of mine.', 'pos'),

    ("I can't believe I'm doing this.", 'neg')

]
from textblob.classifiers import NaiveBayesClassifier

cl = NaiveBayesClassifier(train)
cl.classify("This is an amazing library!")
prob_dist = cl.prob_classify("This one's a doozy.")

prob_dist.max(), round(prob_dist.prob("pos"), 2), round(prob_dist.prob("neg"), 2)
blob = TextBlob("I lost the battle. But I won the war. Happy ending? Maybe!", classifier=cl)

blob.classify()
for s in blob.sentences:

    print(s)

    print(s.classify())
cl.accuracy(test)
cl.show_informative_features(10) # Recall that these are the top word features from our original training set
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()



vs = sid.polarity_scores("Vader is a cool sentiment analyzer that was built for social media."

                         "Exclamations connote positive sentiment!"

                         "More exclamations mean more positivity!!!!!!" #try adding more '!' to this line and rerunning this cell

                         "Is this a problematic assumption?")

print(vs)
def get_vader_score(text):

    # Polarity score returns dictionary

    sentences = nltk.tokenize.sent_tokenize(text)

    for sent in sentences:

        ss = sid.polarity_scores(sent)

        for k in sorted(ss):

            print('{0}: {1}, '.format(k, ss[k]), end='')

            print()

        

get_vader_score(text)
get_vader_score(text), get_textBlob_score(text)