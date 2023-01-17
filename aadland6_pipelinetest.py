import nltk

import re

import json

import string

from nltk import tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords, wordnet

from nltk import pos_tag

from pkg_resources import resource_filename as filename

DEFAULT_STOPWORDS = set(stopwords.words("english")) | set(string.ascii_letters)

DEFAULT_LEMMATIZER = WordNetLemmatizer()





#### Data Setup ####

DEFAULT_STRING = "@#$%^&*()+=,.:;'{}[]|<>`?“”"

EXCLUSION = {}

for character in DEFAULT_STRING:

    EXCLUSION[character] = ""

EXCLUSION['"'] = ""

EXCLUSION["\\"] = " "

EXCLUSION["/"] = " "

EXCLUSIONS_TABLE = EXCLUSION

# negations map 

NEGATIONS = ["not ", "no ",]

NEGATIONS_MAP = ["not_", "no_"]

NEGATIONS_TABLE = {}

for negation, conversion in zip(NEGATIONS, NEGATIONS_MAP):

    NEGATIONS_TABLE["\\b{0}\\b".format(negation)] = conversion

NEGATIONS = NEGATIONS_TABLE



NEGATIONS_RE = re.compile("{0}".format("|").join(NEGATIONS.keys()),

                          flags=re.IGNORECASE)



#### Processing Functions ####

def replacement_gen(document, repl_dict=NEGATIONS, repl=NEGATIONS_RE):

    """ Replaces specific phrases with a corresponding term

        Args:

            document(str): pre-tokenized string

            repl_dict(dict): dict of words and their replacements

            repl(SRE_Pattern): precompiled regrex pattern

        Returns:

            document(str): orginal document with the target words replaced

    """

    def replace(match):

        """ replaces the match key

        """

        match_token = "\\b{0}\\b".format(match.group(0).lower())

        return repl_dict[match_token]

    return repl.sub(replace, document)

def token_gen(document):

    """Generates tokens using nltk.word_tokenize

        Args:

            document(str): higher level document primative

        Returns:

            tokens(list): list of word tokens

    """

    return tokenize.word_tokenize(document)



# def keep_gen(token, LETTERS=string.ascii_letters):

#    return [token for token in tokens if set(tokens).intersection(LETTERS)]



def clean_gen(tokens, exclusion_table=EXCLUSIONS_TABLE, LETTERS=set(string.ascii_letters)):

    """Cleans a list of tokens

        Args:

            tokens(list): list of word tokens

            exclusion_table(dict) characters to remove from a token

                defaults to "!@#$%^&*()+=,.:;'{}[]|<>`?“”"

        Returns:

            clean_tokens(list): tokens with the offending characters removed

    """

    exclusion = str.maketrans(exclusion_table)

    return [token.translate(exclusion).lower() for token in tokens if set(token).intersection(LETTERS)]





def wordnet_get(tagged_tokens):

    """Helper function for normalizing wordnet labels

    """

    out_tokens = []

    for token in tagged_tokens:

        if token[1].startswith("J"):

            out_token = (token[0], wordnet.ADJ)

        elif token[1].startswith("V"):

            out_token = (token[0], wordnet.VERB)

        elif token[1].startswith("R"):

            out_token = (token[0], wordnet.ADV)

        else:

            out_token = (token[0], wordnet.NOUN)

        out_tokens.append(out_token)

    return out_tokens



def pos_gen(tokens):

    """Generates parts of speech and normalizes them to wordnet labels

    """

    tagged = wordnet_get(pos_tag(tokens))

    return tagged 



def lemma_gen(tokens, wnl=DEFAULT_LEMMATIZER, tag=False):

    """Lemmatizes words

        Args:

            tokens(list): list of word strings

            wnl(WordNetLemmatizer): lemmatizer object

            tag(bool): performs part of speech tagging if true defaults to False 

                for speed

        Returns:

            lemms(list):list of words that have been lemmatized

    """

    if tag:

        wnl_tokens = pos_gen(tokens)

        lemmas = [wnl.lemmatize(token[0], pos=token[1]) for token in wnl_tokens]

    else:

        lemmas = [wnl.lemmatize(token) for token in tokens]

    return lemmas

def stopword_gen(tokens, default=DEFAULT_STOPWORDS, custom=None):

    """Removes the stopwords

        Args:

            tokens(list): list of word tokens

            default(set): set of the stopwords in nltk's stopword corpus

            custom(set): custom stopwords

        returns:

            no_stops(list): lists of words not found in either set

    """

    if custom is not None:

        module_stopwords = default | custom

    else:

        module_stopwords = default

    return [word for word in tokens if word not in module_stopwords]

def default_gen(documents):

    """Default pipeline for cleaning a text field

        Args:

            documents(list): list of strings with the top level document

            Runs in the order of:

                1. Tokenize

                2. Character level cleaning (numbers, punctuation, etc.)

                3. Part of Speech Tagging

                4. Lemmatize the tokens

                5. Generate phrases for negations

                6. Remove stopwords

        Yields:

            finished_document(list): list of tokens with the text normalized

    """

    for document in documents:

        tokens = token_gen(document)

        clean_tokens = clean_gen(tokens)

        lemmas = lemma_gen(clean_tokens, tag=True)

        # add the phrase model here

        negations = replacement_gen(" ".join(lemmas), NEGATIONS,

                                    NEGATIONS_RE)

        new_tokens = negations.split(" ")

        finished_document = stopword_gen(new_tokens)

        yield finished_document

def pre_phrases(documents):

    """Generate a pipeline before the phrases are built

    """

    for document in documents:

        doc_string = " ".join(document)

        raw_tokens = token_gen(doc_string)

        clean_tokens = clean_gen(raw_tokens)

        lemmas = lemma_gen(clean_tokens, tag=False)

        yield lemmas
import pandas as pd

from gensim.models import Phrases

from gensim.models.phrases import Phraser

from sklearn.model_selection import train_test_split as split

from sklearn.metrics import classification_report

from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter

from random import shuffle

data = pd.read_json("../input/train.json")

#### Function Pipeline for the Data processing ####

def ingredient_pipeline(recipe, bigram_model):

    """Operates on individual recipes

    """

    new_recipe = []

    for ingredient in recipe:

        ingredient_tokens = token_gen(ingredient)

        clean_tokens = clean_gen(ingredient_tokens)

        lemmas = lemma_gen(clean_tokens)

        bigrams = bigram_model[lemmas]

        clean_ingredient = " ".join(bigrams)

        new_recipe.append(clean_ingredient)

    return new_recipe



def all_recipes(recipes, bigram_model):

    """Operates on all the recipes

    """

    for recipe in recipes:

        yield ingredient_pipeline(recipe, bigram_model)



def subset_pairs(X_rows, y_rows, target_1, target_2):

    """Generates a unique classifier to distinguish between two cuisine types 

    """

    X_index, y_subset = [], []

    for index, row in enumerate(y_rows):

        if row == target_1 or row == target_2:

            y_subset.append(row)

            X_index.append(index)

    X_subset = X_rows[X_index]

    return X_subset, y_rows[y_subset]



def regroup(X_rows, y_rows, target_1, target_2):

    """Splits the testing pairs again

    """

    X_index, y_subset = [], []

    for index, row in enumerate(y_rows.index):

        if y_rows[row] == target_1 or y_rows[row] == target_2:

            y_subset.append(row)

            X_index.append(index)

    X_subset = X_rows[X_index]

    return X_subset, y_rows[y_subset]



def pairwise_clf(X_predictions, y_predictions, target_1, target_2):

    """Generates a binary classifier between target_1 and target_2

    """

    retrain_x, retrain_y = regroup(X_train, y_train, target_1, target_2)

    retrain_test_x, retrain_test_y = regroup(X_test, y_predictions, target_1, target_2)

    retrain_clf = SGDClassifier(shuffle=True).fit(retrain_x, retrain_y)

    retrain_predictions = retrain_clf.predict(retrain_test_x)

    retrain_predictions = pd.Series(retrain_predictions, index=retrain_test_x.index)

    # print(classification_report(retrain_test_y, retrain_predictions))

    return retrain_predictions



def reassign_groups(X_train, y_train, first_pred, y_test, targets):

    """Reassigns groups through an iterator

    """

    reassigned_predictions = first_pred

    for target in targets:

        print("Converting {0} and {1}".format(target[0], target[1]))

        prediction_next = pairwise_clf(X_train, y_train, target[0], target[1])

        reassigned_predictions[prediction_next.index] = prediction_next

        #print(classification_report(y_test, reassigned_predictions))

    return reassigned_predictions
text_gen = pre_phrases(data["ingredients"])

bigram_model = Phraser(Phrases(text_gen))

recipe_generator = all_recipes(data["ingredients"], bigram_model)

recipe_list = [x for x in recipe_generator]

tfidf_vectorizer = TfidfVectorizer(tokenizer = lambda doc: doc, lowercase=False)

matrix = tfidf_vectorizer.fit_transform(data["ingredients"])

print(matrix.shape)
X_train, X_test, y_train, y_test = split(matrix, data["cuisine"], test_size=.20)

sgd_clf = SGDClassifier(shuffle=True).fit(X_train, y_train)

sgd_predictions = sgd_clf.predict(X_test)

sgd_series = pd.Series(sgd_predictions, index=y_test.index)

sgd_report = classification_report(y_test, sgd_predictions)

print(sgd_report)
pairs = [("vietnamese", "thai"), ("brazilian", "mexican"), ("cajun_creole", "southern_us"),

        ('french', 'italian'), ('greek', 'italian'), ('british', 'southern_us'), 

        ('southern_us', 'italian'), ('southern_us', 'mexican'), ('irish', 'southern_us'),

        ('british', 'italian'), ('italian', 'mexican'), ('russian', 'french'), 

         ('russian', 'southern_us'),

        ('spanish', 'italian'), ('filipino', 'chinese')]

new_predictions = reassign_groups(X_train=X_train, y_train=y_train, first_pred=sgd_series, y_test=y_test,

                                 targets=pairs)
test_data = pd.read_json("../input/test.json")

test_generator = all_recipes(test_data["ingredients"], bigram_model)

test_list = [x for x in test_generator]

test_matrix = tfidf_vectorizer.transform(test_list)



print(test_matrix.shape)
inital_test = sgd_clf.predict(test_matrix)

inital_series = pd.Series(inital_test, index=test_data["ingredients"].index)

X_test = test_matrix

final_predictions = reassign_groups(X_train=X_train, y_train=y_train, first_pred=inital_series, y_test=y_test,

                                 targets=pairs)
id = inital_series.index

cuisine = final_predictions

submission = pd.DataFrame({"id":id, "cuisine":cuisine})

print(submission.head())

print(id[0:100])
import os

print(os.listdir())