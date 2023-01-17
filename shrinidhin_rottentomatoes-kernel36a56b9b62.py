# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup

import re

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import zipfile

with zipfile.ZipFile('../input/sentiment-analysis-on-movie-reviews/train.tsv.zip', 'r') as zip_ref_t:

    zip_ref_t.extractall('../output')

with zipfile.ZipFile('../input/sentiment-analysis-on-movie-reviews/test.tsv.zip', 'r') as zip_ref_te:

    zip_ref_te.extractall('../output')
train_data=pd.read_csv('../output/train.tsv',delimiter='\t')

test_data=pd.read_csv('../output/test.tsv',delimiter='\t')
train_data['Phrase'][0]
train_data.head()
def process_phrase(ph):

    #Using Regular Expressions to further process the string

    process = re.sub("[^a-zA-Z?!.;:]", # The pattern to search for

                      " ",                   # The pattern to replace it with

                      ph)  # The text to search

    

    #We will convert the string to lowercase letter and divide them into words

    words=ph.lower().split()

        

    #Searching a set is much faster than searching list, so we will convert the stop words into a set

    stops = set(stopwords.words("english")) 

    

    #We now remove the stop words or the unimportant words and retain only meaningful ones

    mean_words=[w for w in words if not w in stops]

    return " ".join(mean_words)
#Processing Each Phrase

train_data['Phrase']=[process_phrase(p) for p in train_data['Phrase']]
#Implementing BOW Model

vectorizer=CountVectorizer(analyzer='word',

                         tokenizer=None,

                         preprocessor=None,

                         stop_words=None,

                         max_features=5000)

train_data_features=vectorizer.fit_transform(train_data['Phrase'])
#vectorizer.get_feature_names()
train_data_features.shape
from sklearn.linear_model import LogisticRegression

log_rot=LogisticRegression()

log_rot.fit(train_data_features,train_data['Sentiment'])
test_data.head()
#Pre-processing Test Data

test_data['Phrase']=[process_phrase(p) for p in test_data['Phrase']]
test_data_features=vectorizer.transform(test_data['Phrase'])

test_data_features=test_data_features.toarray()
results=log_rot.predict(test_data_features)
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100)
forest.fit(train_data_features,train_data['Sentiment'])
fores_results=forest.predict(test_data_features)
#Saving submissions

output_file=pd.DataFrame(data={'PhraseID':test_data['PhraseId'],'Sentiment':fores_results})

output_file.to_csv('mysubmissions.csv',index=False)

print('The submission file has been saved successfully')
output_file.head()