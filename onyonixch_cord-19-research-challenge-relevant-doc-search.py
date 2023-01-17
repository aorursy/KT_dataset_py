import numpy as np

import pandas as pd

import spacy



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith(".json"):

            break

        print(os.path.join(dirname, filename))

#You can change these terms for your own paper search:

terms = ["support", "care facilities", "nurse facilities",]
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv")

meta.head()
#Looking for NaN's in meta["title"]:

print(f'There are {meta["title"].isna().sum()} NaN values in meta[title]')



#Looking for NaN's in meta["abstract"]:

print(f'There are {meta["abstract"].isna().sum()} NaN values in meta[abstract]')

print( "-" * 50)



#Check for rows where "title" and "abstract" is NaN:

na = meta["title"].isna() & meta["abstract"].isna() 



#select only the rows where at least the title is there:

   

meta_clean = meta[~na]



print("\n\nData after cleaning:")

print( "-" * 50)

print(f'There are {meta_clean["title"].isna().sum()} NaN values in meta_clean[title]')

print(f'There are {meta_clean["abstract"].isna().sum()} NaN values in meta_clean[abstract]')
#Import NLP librarie spacy

import spacy

from spacy.matcher import PhraseMatcher

from collections import defaultdict



#Generate model

nlp = spacy.load('en')

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')



term_search = defaultdict(list)
#iterate over doc titels in meta data:



patterns = [nlp(text) for text in terms]

matcher.add("SearchTerms", None, *patterns)



for idx, meta in meta_clean.iterrows():

    doc = nlp(meta.title)

    matches = matcher(doc)



    found_items = set([doc[match[1]:match[2]] for match in matches])



    for item in found_items:

        term_search[str(item).lower()].append(meta.title)
for term in terms:

    print(f'{len(term_search[term])} papers found for term -{term}-')
term_search["care facilities"]