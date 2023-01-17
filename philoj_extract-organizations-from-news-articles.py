# Download spaCy pretrained statistical models for English (also silence shell output)

!python -m spacy download en_core_web_lg > /dev/null

import spacy

# Load spCy pretrained statistical models for English

nlp = spacy.load("en_core_web_sm")



import itertools

from pprint import pprint

import pandas as pd

df = pd.read_csv("../input/all-the-news/articles1.csv")



# Companies, agencies, institutions, etc. are classified under the label 'ORG':

# (see: https://spacy.io/api/annotation#named-entities)

ORGANIZATION_LABEL = 'ORG'



# Original dataset has thousands of rows.

# Slice the iterator upto the maximum row count:

MAX_ROW_COUNT = 10

selected_articles = itertools.islice(df.itertuples(), MAX_ROW_COUNT)



# The content is in the 'content' coloumn of each row

article_contents = map(lambda row: row.content, selected_articles)



# Print all organization names in alphabetical order:

pprint(

    sorted(list(set(X.text for content in article_contents for X in nlp(content).ents  if X.label_ == ORGANIZATION_LABEL)))

)