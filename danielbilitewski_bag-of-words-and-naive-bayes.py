import pandas as pd

import numpy as np



# import spacy for NLP and re for regular expressions

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

import re



# import sklearn transformers, models and pipelines

from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, RandomizedSearchCV



# import distributions for randomized grid search

from scipy.stats import uniform, randint



# Load the small language model from spacy

nlp = spacy.load('en_core_web_sm')



# set pandas text output to 400

pd.options.display.max_colwidth = 400
# load data

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



# print shape of datasets

print('Train shape: {}'.format(train.shape))

print('Test shape: {}'.format(test.shape))

print('Sample submission shape: {}'.format(sample_submission.shape))



# inspect train set

train.head()
# find duplicate rows with same text and target, keep only the first

train.drop_duplicates(subset = ['text', 'target'], inplace = True)



# some rows have the same text, but different targets

# drops all of these rows

train.drop_duplicates(subset = 'text', keep = False, inplace = True)



# print new shape of train set

print('Train shape: {}'.format(train.shape))
# create machine learning pipeline

nb_pipe = make_pipeline(

    CountVectorizer(),

    MultinomialNB())
# create train set, test set and target

X_train = train.text

X_test = test.text

y_train = train.target
# cross validate

print('F1 score: {:.3f}'.format(np.mean(cross_val_score(nb_pipe, X_train, y_train, scoring = 'f1'))))



# fit pipeline

nb_pipe.fit(X_train, y_train)



# predict on test set

pred = nb_pipe.predict(X_test)



# submit prediction

sample_submission.target = pred

sample_submission.to_csv('naive_bayes_baseline.csv', index = False)
def tokenize(string, stop_words):

    """

    Tokenize a document passed as a string, remove stop words and 

    return all tokens as a single document in the same order.

    """

    

    # Create a document object

    doc = nlp(string)



    # Generate tokens

    tokens_with_stopwords = [token.text for token in doc]

    

    # remove stop words

    tokens = [token for token in tokens_with_stopwords if token not in stop_words]



    # Convert tokens into a string and return it

    return ' '.join(tokens)



def lemmatize(string):

    """

    Lemmatize a document passed as a string and return all lemmas as a document in the same order.

    """

    # Create a document object

    doc = nlp(string)



    # Generate tokens

    lemmas = [token.lemma_ for token in doc]



    # Convert tokens into a string and return it

    return ' '.join(lemmas)



# tokenize the train and test set

X_train = X_train.apply(tokenize, stop_words = STOP_WORDS)

X_test = X_test.apply(tokenize, stop_words = STOP_WORDS)



# lemmatize the train and test set

X_train = X_train.apply(lemmatize)

X_test = X_test.apply(lemmatize)



# create target

y_train = train.target.copy()
# cross validate

print('F1 score: {:.3f}'.format(np.mean(cross_val_score(nb_pipe, X_train, y_train, scoring = 'f1'))))



# fit pipeline

nb_pipe.fit(X_train, y_train)



# predict on test set

pred = nb_pipe.predict(X_test)



# submit prediction

sample_submission.target = pred

sample_submission.to_csv('naive_bayes_spacy_pipeline.csv', index = False)
def tokenize(string, stop_words):

    """

    Tokenize a document passed as a string, remove stop words and 

    return all tokens as a single document in the same order.

    """

    

    # Create a document object

    doc = nlp(string)



    # Generate tokens

    tokens_with_stopwords = [token.text for token in doc]

    

    # remove stop words

    tokens = [token for token in tokens_with_stopwords if token not in stop_words]



    # Convert tokens into a string and return it

    return ' '.join(tokens)



def preprocess(series):

    """

    Function to clean the tweets by replacing words or characters with little meaning.

    

    Replaces all hyperlinks, numbers, mentions and hashtags with a single identifier 

    (e.g. 'https://google.com' becomes 'HYPERLINK')

    

    Replaces special characters such as exclamation marks, question marks, quotation marks and brackets.

    

    Replaces double or more white spaces with a single white space.

    """

    # replace all hyperlinks

    series = series.map(lambda string: re.sub(r'http.*', 'HYPERLINK', string))



    # replace all numbers

    series = series.map(lambda string: re.sub(r'[0-9,.:]+', 'NUMBER', string))



    # replace all mentions

    series = series.map(lambda string: re.sub(r'@\w+', 'MENTION', string))



    # replace all hashtags

    series = series.map(lambda string: re.sub(r'#', 'HASHTAG', string))



    # replace all symbols

    series = series.map(lambda string: re.sub(r"[\!\?\'\"\{\[\(\)\]\}]", '', string))



    # replace all double space or more with a single space

    series = series.map(lambda string: re.sub(r'[ ][ ]+', ' ', string))

    

    # return series

    return series



# tokenize the text

X_train = train.text.apply(tokenize, stop_words = STOP_WORDS)

X_test = test.text.apply(tokenize, stop_words = STOP_WORDS)



print('Tokenized tweets: --------------------\n')

print(X_train)



# preprocess the train and test set

X_train = preprocess(X_train)

X_test = preprocess(X_test)



print('\nPreprocessed tweets: --------------------\n')

print(X_train)



# lemmatize the train and test set

X_train = X_train.apply(lemmatize)

X_test = X_test.apply(lemmatize)



print('\nLemmatized preprocessed tweets: --------------------\n')

print(X_train)



# create target

y_train = train.target.copy()
# cross validate

print('F1 score: {:.3f}'.format(np.mean(cross_val_score(nb_pipe, X_train, y_train, scoring = 'f1'))))



# fit pipeline

nb_pipe.fit(X_train, y_train)



# predict on test set

pred = nb_pipe.predict(X_test)



# submit prediction

sample_submission.target = pred

sample_submission.to_csv('naive_bayes_custom_pipeline.csv', index = False)
# create a parameter grid

param_distributions = {

    'countvectorizer' : [CountVectorizer(), TfidfVectorizer(max_df = 0.8)],

    'countvectorizer__ngram_range' : [(1,1), (1,2), (1,3)],

    'countvectorizer__min_df' : [1, 2, 3],

    'multinomialnb__alpha' : uniform(loc = 0.7, scale = 0.3)

}



# create a RandomizedSearchCV object

nb_random_search = RandomizedSearchCV(

    estimator = nb_pipe,

    param_distributions = param_distributions,

    n_iter = 200,

    scoring = 'f1',

    n_jobs = -1,

    refit = True,

    verbose = 1,

    random_state = 164,

    return_train_score = True

)



# fit RandomizedSearchCV object

nb_random_search.fit(X_train, y_train)



# print grid search results

cols = ['param_countvectorizer', 

        'param_countvectorizer__min_df', 

        'param_countvectorizer__ngram_range', 

        'param_multinomialnb__alpha', 

        'mean_test_score', 

        'mean_train_score']



pd.options.display.max_colwidth = 50



nb_random_search_results = pd.DataFrame(nb_random_search.cv_results_).sort_values(by = 'mean_test_score', 

                                                                                  ascending = False)

nb_random_search_results[cols].head(10)
# predict on test set with the best model from the randomized search

pred = nb_random_search.predict(X_test)



# submit prediction

sample_submission.target = pred

sample_submission.to_csv('naive_bayes_tuned.csv', index = False)