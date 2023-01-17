import pandas as pd

import os

import json
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
df_info = pd.read_csv('../input/ted-talks-transcript/tedDirector.stats.csv',

                      names=['lang','number'],sep='\t')
LANGUAGES = list(df_info.lang)

print(sorted(LANGUAGES))
print('Total number of langauges :', len(LANGUAGES))
# Metadata from YOUTUBE

df_metadata = pd.read_csv('../input/ted-talks-transcript/ted_metadata_youtube.csv')

df_metadata.info()
df_metadata.head(3)
# Sample the titles:

df_metadata.fulltitle.values
df_kaggle = pd.read_csv('../input/ted-talks-transcript/ted_metadata_kaggle.csv')

df_kaggle.info()
df_kaggle.head(3)
import re

import nltk
import string

from nltk.corpus import stopwords # Common stopwords 

from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer



### Functions for simple preprocessing of summary:

sw = stopwords.words('english')

sw.extend(list(string.punctuation))

stop = set(sw)



# tokenizing the sentences; this uses punkt tokenizer

tokenizer = RegexpTokenizer(r'\w+')

tokenize = lambda x : tokenizer.tokenize(x)



# apply stopping, and remove tokens that have length of 1

removeSW = lambda x : list([t.lower() for t in x if t.lower() not in stop and len(t) > 1 and t.isalpha()])



# Lemmatizing

lemmatizer = WordNetLemmatizer()

lemmify = lambda x : [lemmatizer.lemmatize(t) for t in x]



preprocess = lambda x: lemmify(removeSW(tokenize(x)))
# Approach 2): lower case both df for comparison:

df_kaggle['title_compare'] = df_kaggle['name'].apply(lambda x:preprocess(x))

df_metadata['fulltitle_compare'] = df_metadata['fulltitle'].apply(lambda x :preprocess(x))
df_metadata['fulltitle_compare'].iloc[0:5].values
df_kaggle['title_compare'].iloc[0:5].values
metadata_compare = df_metadata[['fulltitle_compare', 'id']] # small dataset for comparing

metadata_compare.set_index('id',inplace=True)
def find_match(toks_kaggle, threshold=0.8):

    #     only accept the vidID if the threshold is greater than or equal 0.8   

    compare = metadata_compare.copy()

    compare['ratings'] = compare['fulltitle_compare'].apply(lambda meta: compute_score(toks_kaggle, meta))

    vidID = compare.ratings.idxmax()

    

    if compare.ratings.max() >= threshold:

#         logging.info("FOUND: {}".format(compare.loc[vidID]))

#         logging.info("-----ACCEPT----")

        return vidID

    elif compare.ratings.max() >= .7:

        # Use this to check if the threshold is ``enough''

        logging.info("KAGGLE TITLE: {}".format(toks_kaggle))

        logging.info("FOUND: {} ({})".format(compare.loc[vidID].fulltitle_compare, compare.loc[vidID].ratings))

        logging.info("-----REJECT----")

        return ""

    else:

        return ""



def compute_score(kaggle, meta):

    total= len(kaggle)

    count = 0

    for toks in meta:

        if toks in kaggle: 

            count += 1

    return count/total
# This would generate the youtube Video ID for the kaggle dataset

df_kaggle['vidID_youtube'] = df_kaggle['title_compare'].apply(lambda x : find_match(x))
# Proportion of kaggle dataset not labelled with any videoIDs:

print(list(df_kaggle.vidID_youtube.values).count(''), "/", len(df_kaggle), " = ", list(df_kaggle.vidID_youtube.values).count('')/len(df_kaggle))
## These are then save for this dataseet.

# df_kaggle.to_csv('./ted_metadata_kaggle.csv')

# df_metadata.to_csv('./ted_metadata_youtube.csv')