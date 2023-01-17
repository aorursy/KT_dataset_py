# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/competition-06-dataset/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import spacy

from spacy.lang.en import English

from spacy.lang.en.stop_words import STOP_WORDS

import re
train_df = pd.read_csv("/kaggle/input/competition-06-dataset/train_nlp.csv")

test_df = pd.read_csv("/kaggle/input/competition-06-dataset/test_nlp.csv")

sampleSubmission_df = pd.read_csv("/kaggle/input/competition-06-dataset/sampleSubmission.csv")
train_df.head()
test_df.head()
sampleSubmission_df.head()
dataset = train_df#.sample(n=50, replace=False, random_state=42)
def deEmojify(text):

    regrex_pattern = re.compile(pattern = "["

        u"\U0001F600-\U0001F64F"  # emoticons

        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

        u"\U0001F680-\U0001F6FF"  # transport & map symbols

        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           "]+", flags = re.UNICODE)

    return regrex_pattern.sub(r'',text)
dataset['review'] = dataset['review'].apply(deEmojify)
# Load English tokenizer, tagger, parser, NER and word vectors

nlp = spacy.load("en_core_web_sm")



def word_tokenizer(text):

    my_doc = nlp(text)

    token_list = []

    for token in my_doc:

        token_list.append(token.text)

    return(token_list)
dataset['Word Tokenized Review'] = dataset['review'].apply(word_tokenizer)
# nlp = spacy.load("en_core_web_sm")



# def sentence_tokenizer(text):

#     # Create the pipeline 'sentencizer' component

#     sbd = nlp.create_pipe('sentencizer')

#     # Add the component to the pipeline

#     nlp.add_pipe(sbd)

#     #  "nlp" Object is used to create documents with linguistic annotations.

#     doc = nlp(text)

#     # create list of sentence tokens

#     sents_list = []

#     for sent in doc.sents:

#         sents_list.append(sent.text)

#     return(sents_list)
# dataset['Sentence Tokenized Review'] = dataset['review'].apply(sentence_tokenizer)
dataset.head()
def stop_words_removal(text):

    filtered_value = []

    spacy_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)

    for word in text:

        if not word in spacy_stopwords:

            filtered_value.append(word)

    return(filtered_value)
dataset['Word Tokenized Review w/o StopWords'] = dataset['Word Tokenized Review'].apply(stop_words_removal)
nlp = spacy.load("en_core_web_sm")

def alemantiser(text):

    alist = []

    for word in text:

        sometext = nlp(word)

        for avalue in sometext:

            lem_text = avalue.lemma_

            alist.append(lem_text)

    return alist
dataset['Word Tokenized Review w/o StopWords w Lemantised'] = dataset['Word Tokenized Review w/o StopWords'].apply(alemantiser)
def join_back(x):

    separator = ' '

    return(separator.join(x).strip())



dataset['Joined Cleaned Review'] = dataset['Word Tokenized Review w/o StopWords w Lemantised'].apply(join_back)
dataset.to_csv("Joined Cleaned Review.csv")
# We just want the vectors so we can turn off other models in the pipeline

with nlp.disable_pipes():

    vectors = np.array([nlp(review['Joined Cleaned Review']).vector for idx, review in dataset.iterrows()])

    

vectors.shape
np.savetxt('train_data_vectors.txt', vectors)
#b = np.loadtxt("/kaggle/input/competition-06-dataset/train_data_vectors.txt")
from sklearn.svm import LinearSVC



y_train = train_df.rating[:200]



# Create the LinearSVC model

model = LinearSVC(random_state=1, dual=False, max_iter=10000)

# Fit the model

model.fit(vectors, y_train)
# Scratch space in case you want to experiment with other models

import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)



xg_reg.fit(vectors, y_train)
test_df["rating"] = model.predict(test_df.review)
test_df["rating"] = xg_reg.predict(review)