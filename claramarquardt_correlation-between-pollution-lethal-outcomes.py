# Install dependencies

! pip install spacy_langdetect



# Import libraries

import os

import pandas as pd

import numpy as np

import pickle

import re



import spacy

from spacy_langdetect import LanguageDetector

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation
# Load the data - pre-processed titles & abstracts + full raw data

## Source: https://www.kaggle.com/skylord/coronawhy

df_covid      = pd.read_csv("/kaggle/input/coronawhy/titles_abstracts_processed_03282020.csv")

df_covid_meta = pd.read_csv("/kaggle/input/coronawhy/clean_metadata.csv")
# Inspect the data

print(df_covid.head)

print(df_covid.columns) 
# Define keywords

## Source: https://trello.com/c/BaeJaGWo/50-keywords-for-pollution & https://trello.com/c/1BfTSzHg/51-keywords-for-lethal-outcomes



## Define raw keywords

pollution_key = ["pollution", "contamination", "contaminating"]

lethal_key    = ["death", "fatality", "mortality","lethal", "lethality", "morbidity"]



## Lemmatize keywords

### Initialize spacy model

nlp = spacy.load('en')



### Lemmatize keywords

pollution_key    = [nlp(x)[0] for x in pollution_key]

pollution_key    = np.unique([x.lemma_ for x in pollution_key])

print(pollution_key)



lethal_key = [nlp(x)[0] for x in lethal_key]

lethal_key = np.unique([x.lemma_ for x in lethal_key])

print(lethal_key)



### Create regex patterns

pollution_key_pattern = r"\b" + r"\b|\b".join(pollution_key) + r"\b"

print(pollution_key_pattern)

lethal_key_pattern    = r"\b" + r"\b|\b".join(lethal_key) + r"\b"

print(lethal_key_pattern)
# Identify co-occurence of keywords in the data



## Identify sentences which mention both a risk & lethal outcome keyword (lemmatized form)

df_covid_key = df_covid.loc[[bool(re.search(r"\b"+pollution_key_pattern+r"\b",x,re.IGNORECASE)) 

                                 & bool(re.search(r"\b"+lethal_key_pattern+r"\b",x,re.IGNORECASE)) for x in df_covid.lemma]]
# Inspect and save the results



## Inspect the results

print(str(len(df_covid_key)) + " unique sentences found")

print(str(len(set(df_covid_key._id))) + " unique articles found\n\n")



## Validate the specific sentences found

pd.set_option('display.max_colwidth', -1)

print(df_covid_key.lemma[0:10])



## Pivot the results (one column per 'match')

df_covid_key.loc[:,('id')] = df_covid_key.groupby(['_id']).cumcount()+1

df_covid_key = df_covid_key.pivot_table(index=['_id'],columns='id', values='lemma', aggfunc='first').reset_index()

column_name = ["match_"+str(x) for x in np.arange(0,len(df_covid_key.columns)-1)]

df_covid_key.columns.values[1:] = column_name



## Merge in the abstracts

df_covid_key_abs = df_covid_key.merge(df_covid_meta[["sha","title","abstract"]], left_on='_id', right_on='sha', how='left')



## Save the results

df_covid_key_abs.to_excel("df_covid_key_final.xlsx")