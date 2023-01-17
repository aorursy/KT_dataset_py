# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup



from spacy.lang.en import English

nlp = English()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load Data

print("Loading data...")

train = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)

train = train.drop(['article_link'], axis=1)



print("Train shape:", train.shape)

train.head()
# Check the first review



print('The first review is:\n\n',train["headline"][0])
# function to clean data



def cleanData(doc,stemming = False):

    doc = doc.lower()

    doc = nlp(doc)

    tokens = [tokens.lower_ for tokens in doc]

    tokens = [tokens for tokens in doc if (tokens.is_stop == False)]

    tokens = [tokens for tokens in tokens if (tokens.is_punct == False)]

    final_token = [token.lemma_ for token in tokens]

    

    return " ".join(final_token)
clean_review = cleanData(train['headline'][0])

clean_review
# clean description

print("Cleaning train data...\n")

train["headline"] = train["headline"].map(lambda x: cleanData(x))
sample_text = """When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously. “I can tell you very senior CEOs of major American car companies would shake my hand and turn away because I wasn’t worth talking to,” said Thrun, now the co-founder and CEO of online higher education startup Udacity, in an interview with Recode earlier this week.



A little less than a decade later, dozens of self-driving startups have cropped up while automakers around the world clamor, wallet in hand, to secure their place in the fast-moving world of fully automated transportation."""

doc = nlp(sample_text)
# print each token

for token in doc:

    print(token.text)
import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(sample_text)
for token in doc:

    # Print the text and the predicted part-of-speech tag

    print(token.text, token.pos_)
# Iterate over the predicted entities

for ent in doc.ents:

    # Print the entity text and its label

    print(ent.text, ent.label_)
# Get quick definitions of the most common tags and labels



print(spacy.explain('GPE'))

print(spacy.explain('ORG'))
spacy.displacy.render(doc, style='ent', jupyter=True)
# Define a custom component

def custom_component(doc):

    # Print the doc's length

    print('Doc length:', len(doc))

    # Return the doc object

    return doc



# Add the component first in the pipeline

nlp.add_pipe(custom_component, first=True)



# Print the pipeline component names

print('Pipeline:', nlp.pipe_names)
# Process a text

doc = nlp(sample_text)
nlp = spacy.load('en_core_web_lg')
doc1 = nlp("My name is shyam")

doc2 = nlp("My name is Ram")
print("The documents similarity is:" ,doc1.similarity(doc2))