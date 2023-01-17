#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
#print(os.listdir("../input"))
import spacy
import random 
from collections import Counter #for counting
import seaborn as sns #for visualization
nlp = spacy.load('en')
tweets = pd.read_csv("../input/justdoit_tweets_2018_09_07_2.csv")
tweets.shape
tweets.head(1)
random.seed(888)
text = tweets.tweet_full_text[random.sample(range(1,100),10)]
text
text_combined = str(text)
doc = nlp(text_combined)
for token in doc:
    print(token)
for token in doc:
    print(token.text, token.pos_)
nouns = list(doc.noun_chunks)
nouns
list(doc.sents)
for ent in doc.ents:
    print(ent.text,ent.label_)
spacy.displacy.render(doc, style='ent',jupyter=True)
for token in doc:
    print(token.text, token.lemma_)
spacy.displacy.render(doc, style='dep',jupyter=True)