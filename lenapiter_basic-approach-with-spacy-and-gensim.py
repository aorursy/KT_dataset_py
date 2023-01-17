# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



from gensim import corpora, models

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd 

import seaborn as sns

from sklearn.naive_bayes import MultinomialNB

import spacy

from spacy import displacy

from spacy.matcher import Matcher

from spacy.tokens import Span

import en_core_web_lg

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv", usecols=[0,1,2])

data.sample(5)
data.target.describe()
sns.distplot(data.target)
data.loc[:,"target_binary"] = np.where(data.target < 0.5, 0, 1)
data.target_binary.value_counts()
# divide data into two dataframes: positive and negative

positive = data[data.target_binary == 0]

positive = positive.sample(100000)



negative = data[data.target_binary == 1]

negative = negative.sample(100000)
# convert content of positive comment_text column to one single string

positive_string = " ".join([word for word in positive.comment_text])

print(len(positive_string))

print(positive_string[:100])
wordcloud = WordCloud(max_font_size=50, background_color="white").generate(positive_string)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# convert content of negative comment_text column to one single string

negative_string = " ".join([word for word in negative.comment_text])

print(len(negative_string))

print(negative_string[:100])
wordcloud = WordCloud(max_font_size=50, background_color="white").generate(negative_string)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
sample = data.sample(1000)
# load spaCy's english large language model

nlp = spacy.load("en_core_web_lg", disable=["ner"])
nlp.pipeline
# instantiate Matcher

matcher = Matcher(nlp.vocab)



# define pattern

pattern = [{"IS_ALPHA": True,

            "IS_STOP": False,

            "LENGTH": {">": 1},

            "LENGTH": {"<=": 20}

           }]



# add pattern to matcher

matcher.add("Cleaning", None, pattern)
# initialize empty list for proccessed texts

texts = []



for idx, row in sample.iterrows():

    # get nlp doc of comment text

    doc = nlp(row.comment_text)

    

    # apply matcher on doc

    matches = matcher(doc)

    

    # initialize empty list for matched tokens

    token_matches = []

    

    for match_id, start, end in matches:

        # add custom entitiy "MATCH" to doc.ents

        doc.ents = list(doc.ents) + [Span(doc, start, end, label="MATCH")]  

    

        # get lemma for matched tokens and write to data frame

        token_matches.append(doc[start:end].lemma_.lower())

        sample.loc[idx, "comment_preprocessed"] = " ".join(token_matches)

    

    # append processed comment to list of texts

    texts.append(token_matches)
displacy.render(doc, style="ent", options={"ents": ["MATCH"]})
sample[["comment_text", "comment_preprocessed"]].sample(10)
dictionary = corpora.Dictionary(texts)
print("The dictionary consists of {} different tokens. In total, {} documents were processed.".format(dictionary.num_pos, dictionary.num_docs))
# get bow representation for each text

corpus_bow = [dictionary.doc2bow(text) for text in texts]



# serialize corpus

corpora.MmCorpus.serialize("corpus.mm", corpus_bow)



# get tfidf representation for each text

corpus_tfidf = models.TfidfModel(corpus_bow)
#for document in corpus_tfidf[corpus_bow]:

#    for token in document:

#        print(token)
for token, id in dictionary.token2id.items():   

    if id == 6007:

        print(token)
# instantiate NB model

clf = MultinomialNB()



# fit classifier on data

clf.fit(corpus_tfidf[corpus_bow], list(sample.target_binary.values))