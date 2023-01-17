import pandas as pd

import numpy as np

import re

import langid



pd.set_option('display.max_colwidth', -1)

pd.set_option("display.max_rows", None)

pd.set_option('display.max_columns', None)



import warnings

warnings.filterwarnings('ignore')
# load data

T = pd.read_csv("../input/all_annotated.tsv", sep = "\t")
T.head()
from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
# write function that returns the identified languages and its confidence scores

# adjust how many languages to store with a threshold parameter 

# that stores only the languages with a confidence score that is no larger than x digits behind the comma.  

def language_confidence(text, threshold=6):

    language_conf = []

    for pair in identifier.rank(text):

        if round(pair[1], threshold) > 0.:

            language_conf.append(pair)    

    return language_conf
T['Tweet'][0:5].apply(identifier.classify)
T['Tweet'][0:5].apply(lambda x: language_confidence(x, threshold=2))
# remove urls starting with http 

http = r'http\S+'

T['Tweet_stripped'] = T['Tweet'].str.replace(http, ' ' )



# remove tags

T['Tweet_stripped'] = T['Tweet_stripped'].apply(lambda x: ' '.join(word for word in x.split(' ')\

                                                                   if not word.startswith('@')))
T['language_conf'] = T['Tweet_stripped'].apply(lambda x: language_confidence(x, threshold=2)) 
# convert country ISO codes to lower case

T['Country'] = T['Country'].str.lower()
# write function that returns the identified language only

def languages(text, threshold=6):

    language_conf = []

    for pair in identifier.rank(text):

        if round(pair[1], threshold) > 0.:

            language_conf.append(pair)    

    return [row[0] for row in language_conf]



T['language_only'] = T['Tweet_stripped'].apply(lambda x: languages(x, threshold=2))
# write function that returns the most likely language(s)

def tweet_language(row):

    if row.Country in row.language_only:

        return row.Country

    elif row.Country == 'us|gb|ie' and 'en' in row.language_only:

        return 'en'

    elif row.Country == 'mx' and 'es' in row.language_only:

        return 'es'

    elif row.Country == 'br' and 'pt' in row.language_only:

        return 'pt'

    elif row.Country == 'br' and 'es' in row.language_only:

        return 'es'

    else:

        return row.language_only

    

T['language'] = T.apply(tweet_language, axis=1)
T[['Country', 'language', 'Tweet_stripped']]