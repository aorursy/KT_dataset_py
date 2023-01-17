# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import string 



import spacy

from spacy.lang.en.stop_words import STOP_WORDS



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import train_test_split



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline





%matplotlib inline

pd.options.display.max_columns = None
test_raw = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

train_raw = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
# From the comments I found that tweets are not unique

# And that sometimes they are mislabeled

print("Total Train Tweets:", len(train_raw['text']))

print("Total Unique Tweets:", len(train_raw['text'].unique()))



duped_tweets = train_raw[train_raw.duplicated(subset=['text'],keep=False)].sort_values(by=['text'])

duped_tweets.head()



train_raw = train_raw[~train_raw.duplicated(subset=['text','target'],keep=False)]

print("Total sans dupes: ", len(train_raw))
# That's all feature engineering we'll do

test_df = test_raw.copy()

train_df = train_raw.copy()
nlp = spacy.load('en_core_web_sm')

stop_words = spacy.lang.en.stop_words.STOP_WORDS

punctuations = string.punctuation
# Creating our tokenizer function

def spacy_tokenizer(sentence):

    # Creating our token object, which is used to create documents with linguistic annotations.

    mytokens = nlp(sentence)



    # Lemmatizing each token and converting each token into lowercase

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    

    # Removing stop words

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]



    # return preprocessed list of tokens

    return mytokens

class predictors(TransformerMixin):

    def transform(self, X, **transform_params):

        # Cleaning Text

        return [clean_text(text) for text in X]



    def fit(self, X, y=None, **fit_params):

        return self



    def get_params(self, deep=True):

        return {}



# Basic function to clean the text

def clean_text(text):

    # Removing spaces and converting text into lowercase

    return text.strip().lower()
# We will use this function to get the best model 

def get_tuned_model(estimator, param_grid, scoring, X, Y):

    from sklearn.model_selection import GridSearchCV



    grid = GridSearchCV(estimator = estimator, 

                       param_grid = param_grid,

                       scoring = scoring,

                       cv=3,

                       n_jobs= -1

                      )



    tuned = grid.fit(X, Y)



    print ("Best score: ", tuned.best_score_) 

    print ("Best params: ", tuned.best_params_)

    print ("IS Score: ", tuned.score(X, Y))

    

    return tuned

def save_results(model, ids, data):

    pred_test = model.predict(data)



    test_res = ids.copy()

    test_res["target"] = pred_test

    test_res.to_csv("/kaggle/working/my_predictions.csv", index=False)

    return test_res
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
X_train, X_valid, y_train, y_valid = train_test_split(train_df['text'], train_df['target'], test_size=0.3)



X_train = tfidf_vector.fit_transform(X_train)

X_valid = tfidf_vector.transform(X_valid)
ids = test_df[['id']]

X_test = tfidf_vector.transform(test_df['text'])
classifier = LogisticRegression()



param_grid = {

    "C":  np.logspace(0, 4, 10),

}



# grd = get_tuned_model(pipe, param_grid, "accuracy", train_df['text'], train_df['target'])

grd = get_tuned_model(classifier, param_grid, "accuracy", X_train, y_train)
results = save_results(grd, ids, X_test)

results.head()