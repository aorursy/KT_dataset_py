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
df_covid      = pd.read_csv("/kaggle/input/coronawhy/fulltext_processed_03282020.csv")
# Inspect the data
print(df_covid.shape)
print(df_covid.head)
print(df_covid.columns) 
# Define keywords

## Define raw keywords (bigrams & trigrams)
## Note - ignoring unigram keywords: CVD, MI, STEMI, NSTEMI, AF, Cardiomyopathy, CAD 
keyword_bigram  = ["cardiac disease", "cerebrovascular disease", "heart disease", "myocardial infarction", 
    "heart failure", "lv dysfunction", "rv dysfunction", "cardiac arrhythmias", "atrial fibrillation",  "a fib"]
keyword_trigram = ["chronic heart diseases", "acute myocardial injury", "coronary artery disease"]

## Lemmatize keywords
### Initialize spacy model
nlp = spacy.load('en')

### Lemmatize keywords & generate regex patterns
keyword_bigram          = [nlp(x) for x in keyword_bigram]
keyword_bigram          = [[x.lemma_ for x in z] for z in keyword_bigram]
keyword_bigram_list     = list(np.unique(sum(keyword_bigram, [])))
keyword_bigram_pattern  = [r"\b" + x + r"\b" for x in keyword_bigram_list]
print(keyword_bigram)
print(keyword_bigram_list)
print(keyword_bigram_pattern)

keyword_trigram         = [nlp(x) for x in keyword_trigram]
keyword_trigram         = [[x.lemma_ for x in z] for z in keyword_trigram]
keyword_trigram_list    = list(np.unique(sum(keyword_trigram, [])))
keyword_trigram_pattern = [r"\b" + x + r"\b" for x in keyword_trigram_list]
print(keyword_trigram)
print(keyword_trigram_list)
print(keyword_trigram_pattern)

keyword_list     = keyword_bigram_pattern + keyword_trigram_pattern
keyword_list_imp = [x for x in keyword_list if len(x)>5]
keyword_pattern = "|".join(keyword_list_imp)
print(keyword_pattern)
# Subset on unigrams

#* Option #1 - Generate df_covid_key
## Subset
print(keyword_pattern)
df_covid_key = df_covid.loc[[bool(re.search(keyword_pattern,x, re.IGNORECASE)) for x in df_covid.lemma]]
print(len(df_covid_key))

## Save
df_covid_key.to_pickle("df_covid_key.pkl")

#* Option #2 - Load previously generated df_covid_key
# df_covid_key = pd.read_pickle("df_covid_key.pkl")
# Generate bigrams / trigrams for subset dataset 

## Extract unigrams
unigram = df_covid_key.lemma

## Generate bigrams
bigram = [[" ".join([z[i],z[i+1]]) for i in range(0,len(z)-1)] for z in unigram.str.split(",")]
print(len(bigram))
print(bigram[1:10])
bigram =  [[x.replace("'","") for x in z] for z in bigram]
bigram =  [[x.replace("[","") for x in z] for z in bigram]
bigram =  [[x.replace("]","") for x in z] for z in bigram]
bigram =  [[x.replace(".","") for x in z] for z in bigram]
bigram =  [[x.replace("(","") for x in z] for z in bigram]
bigram =  [[x.replace(")","") for x in z] for z in bigram]
bigram =  [[x.replace(">","") for x in z] for z in bigram]
bigram =  [[x.replace("<","") for x in z] for z in bigram]
bigram =  [[x.replace("=","") for x in z] for z in bigram]
bigram = [[x.strip() for x in z] for z in bigram]
bigram = [[x for x in z if len(x)>1] for z in bigram]
bigram = [[x for x in z if bool(re.search(" ",x))==True] for z in bigram]
bigram = [[x.replace("  "," ") for x in z] for z in bigram]

## Inspect & save bigrams
print(bigram[1:5])
with open('bigram.pkl', 'wb') as b:
    pickle.dump(bigram,b)

## Generate trigrams
trigram = [[" ".join([z[i],z[i+1],z[i+2] ]) for i in range(0,len(z)-2)] for z in unigram.str.split(",")]
print(len(trigram))
print(trigram[1:10])
trigram =  [[x.replace("'","") for x in z] for z in trigram]
trigram =  [[x.replace("[","") for x in z] for z in trigram]
trigram =  [[x.replace("]","") for x in z] for z in trigram]
trigram =  [[x.replace(".","") for x in z] for z in trigram]
trigram =  [[x.replace("(","") for x in z] for z in trigram]
trigram =  [[x.replace(")","") for x in z] for z in trigram]
trigram =  [[x.replace(">","") for x in z] for z in trigram]
trigram =  [[x.replace("<","") for x in z] for z in trigram]
trigram =  [[x.replace("=","") for x in z] for z in trigram]
trigram = [[x.strip() for x in z] for z in trigram]
trigram = [[x for x in z if len(x)>1] for z in trigram]
trigram = [[x for x in z if bool(re.search(" [a-zA-Z]+ ",x))==True] for z in trigram]
trigram = [[x.replace("  "," ") for x in z] for z in trigram]

## Inspect & save trigrams
print(trigram[1:5])
with open('trigram.pkl', 'wb') as b:
    pickle.dump(trigram,b)

## Check
print(sum([any(['body  weight' in x]) for x in bigram]))
# Find keywords

## Set up keywords
keyword_bigram_match  = [" ".join(x) for x in keyword_bigram]
keyword_trigram_match = [" ".join(x) for x in keyword_trigram]

## Extract rows that contain bigram or trigram
df_covid_key["result"] = 0
for i in keyword_bigram_match:
    print(i)
    result = [any([i in x]) for x in bigram]
    print("Number of sentences that contain this bigram: " + str(sum(result))+ "\n")
    df_covid_key.loc[result, "result"] = 1
for i in keyword_trigram_match:
    print(i)
    result = [any([i in x]) for x in trigram]
    print("Number of sentences that contain this trigram: " + str(sum(result)) + "\n")
    df_covid_key.loc[result, "result"] = 1 
    print(len(df_covid_key.loc[df_covid_key.result==1]))
    
## Inspect
print("\nTotal number of sentences: " + str(len(df_covid_key.loc[df_covid_key.result==1])))
print("Total number of unique articles: " + str(len(np.unique(df_covid_key.loc[df_covid_key.result==1]._id))))
df_covid_key[df_covid_key.result==1][1:100].to_csv("df_covid_key_check.csv")

## Save
df_covid_key[df_covid_key.result==1].to_pickle("df_covid_key_result.pkl")
