import pandas as pd
import numpy as np
import spacy
import seaborn as sns
import random
nlp = spacy.load('en')
tweets = pd.read_csv("../input/justdoit_tweets_2018_09_07_2.csv")
tweets.shape
text = tweets.tweet_full_text[random.sample(range(1,100),10)]
text
combined = str(text)
doc = nlp(combined)
for token in doc:
    print(token)
for token in doc:
    print(token.text,token.pos_)
sents = list(doc.sents)
sents
for entity in doc.ents:
    print(entity.text,entity.label_)
#Entity style Display
spacy.displacy.render(doc, style='ent',jupyter=True)
#Lemmatization
for token in doc:
    print(token.text,token.lemma_)
#Dependency Parser style Display
spacy.displacy.render(doc, style='dep',jupyter=True)
