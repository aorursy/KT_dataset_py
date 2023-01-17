# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

import matplotlib.pyplot as plt

import re

import random

from spacy.util import minibatch, compounding



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/drugsComTrain_raw.csv')

df_train.head()
comment_words = ' '

stopwords = set(STOPWORDS) 

for review in df_train['review']: 

    # typecaste each val to string 

    review = str(review).lower() 

    

    # split the value 

    tokens = review.split()

    comment_words = comment_words + ' '.join(tokens)
wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
df_train.shape
drug_list = df_train['drugName'].value_counts().index.tolist()

drug_list = [x.lower() for x in drug_list]
#First let's check some NERs in first 10 reviews and remove date, time, ordinal and cardinal.

nlp = spacy.load('en_core_web_sm')

count = 0

for review in df_train['review']:

    if count < 11:

        doc = nlp(review)

        ents = [(e.text, e.label_) for e in doc.ents if e.label_ not in ('DATE', 'TIME', 'ORDINAL', 'CARDINAL')]

        print(ents)

    count += 1
def process_review(review):

    processed_token = []

    for token in review.split():

        token = ''.join(e.lower() for e in token if e.isalnum())

        processed_token.append(token)

    return ' '.join(processed_token)
#Step 1: Let's create the training data

count = 0

TRAIN_DATA = []

for _, item in df_train.iterrows():

    ent_dict = {}

    if count < 1000:

        review = process_review(item['review'])

        #We will find a drug and its positions once and add to the visited items.

        visited_items = []

        entities = []

        for token in review.split():

            if token in drug_list:

                for i in re.finditer(token, review):

                    if token not in visited_items:

                        entity = (i.span()[0], i.span()[1], 'DRUG')

                        visited_items.append(token)

                        entities.append(entity)

        if len(entities) > 0:

            ent_dict['entities'] = entities

            train_item = (review, ent_dict)

            TRAIN_DATA.append(train_item)

            count+=1
n_iter = 10

def train():

    nlp = spacy.blank("en")  # create blank Language class

    print("Created blank 'en' model")

    

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    # otherwise, get it so we can add labels

    else:

        ner = nlp.get_pipe("ner")

        

    # add labels

    for _, annotations in TRAIN_DATA:

        for ent in annotations.get("entities"):

            ner.add_label(ent[2])

            

    nlp.begin_training()

    for itn in range(n_iter):

        random.shuffle(TRAIN_DATA)

        losses = {}

        # batch up the examples using spaCy's minibatch

        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))

        for batch in batches:

            texts, annotations = zip(*batch)

            nlp.update(

                texts,  # batch of texts

                annotations,  # batch of annotations

                drop=0.5,  # dropout - make it harder to memorise data

                losses=losses,

            )

        print("Losses", losses)

    return nlp
#Step 2: Let's train custom model with the training data

nlp2 = train()
#Test the model

for text, _ in TRAIN_DATA[:10]:

    doc = nlp2(text)

    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
test_reviews = df_train.iloc[-10:, :]['review']

for review in test_reviews:

    review = process_review(review)

    print(review)

    doc = nlp2(review)

    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    print('________________________')