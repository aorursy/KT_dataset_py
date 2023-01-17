#importing libraries



import spacy

import pandas as pd

import random
nlp = spacy.load('en')
type(nlp)
tweets = pd.read_csv("../input/5000-justdoit-tweets-dataset/justdoit_tweets_2018_09_07_2.csv")
tweets.head()
tweets.shape
tweets['tweet_full_text'][:20]
#random tweets

random.seed(1024)



random_tweets = tweets['tweet_full_text'][random.sample(range(1,5000), 20)]

random_tweets
#combined text



combined_text = str(random_tweets)

combined_text

# len(combined_text)

# type(combined_text)
doc = nlp(combined_text)

doc
type(doc)
#tokenization using split as space

doc.text.split()
[token.orth_ for token in doc]
[(token.orth_, token.orth) for token in doc if not token.is_punct | token.is_space | token.is_stop]
extracted_tokens = [token.orth_ for token in doc if not token.is_punct | token.is_space | token.is_stop]

extracted_tokens
only_word_tokens = [i for i in extracted_tokens if i.isalpha()]

only_word_tokens
list(doc.sents)
[word.lemma_ for word in doc]
pos_tag = [(word, word.tag_, word.pos_) for word in doc]

pos_tag
[i for i in pos_tag if i[1] == 'POS']
[j for j in pos_tag if j[2] == 'PART']
nouns = list(doc.noun_chunks)

nouns
[(token, token.dep_) for token in doc]
[i for i in doc.ents]
[(i, i.label_, i.label) for i in doc.ents]
#named entities along with text labels

spacy.displacy.render(doc, style='ent', jupyter=True)
spacy.displacy.render(doc, style='dep', jupyter=True)