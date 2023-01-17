import os

import json

import re

import pandas as pd

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import FunctionTransformer
# bigrams and typos

def get_replacements():

    return {'wasabe': 'wasabi', '-': '', 'sauc': 'sauce',

            'baby spinach': 'babyspinach', 'coconut cream': 'coconutcream',

            'coriander seeds': 'corianderseeds', 'corn tortillas': 'corntortillas',

            'cream cheese': 'creamcheese', 'fish sauce': 'fishsauce',

            'purple onion': 'purpleonion','refried beans': 'refriedbeans', 

            'rice cakes': 'ricecakes', 'rice syrup': 'ricesyrup', 

            'sour cream': 'sourcream', 'toasted sesame seeds': 'toastedsesameseeds', 

            'toasted sesame oil': 'toastedsesameoil', 'yellow onion': 'yellowonion'}
# return string with all valid ingredients

def tranform_to_single_string(ingredients, lemmatizer, replacements, stop_pattern):

    ingredients_text = ' '.join(iter(ingredients))



    for key, value in replacements.items():

        ingredients_text = ingredients_text.replace(key, value)

    

    words = []

    for word in ingredients_text.split():

        if not stop_pattern.match(word) and len(word) > 2: 

            word = lemmatizer.lemmatize(word)

            words.append(word)

    return ' '.join(words)
def get_estimator():

    return SVC(C=300,

         kernel='rbf',

         gamma=1.5, 

         shrinking=True, 

         tol=0.001, 

         cache_size=1000,

         class_weight=None,

         max_iter=-1, 

         decision_function_shape='ovr',

         random_state=42)
# use this to analize ingredients

def show_unique_ingredients(train):

    ingredients = {}

    for idx, row in train.iterrows():

        for ingredient in row['ingredients']:

            if ingredient not in ingredients:           

                ingredients[ingredient] = {'sum': 0}

            previous = ingredients[ingredient][row['cuisine']] if row['cuisine'] in ingredients[ingredient] else 0

            ingredients[ingredient][row['cuisine']] = 1 + previous

            ingredients[ingredient]['sum'] += 1



    for ingredient in sorted(ingredients):

        for cuisine in sorted(ingredients[ingredient], key=ingredients[ingredient].get, reverse=True):

            print(f'{ingredient}:{cuisine}:{ingredients[ingredient][cuisine]}')
def  preprocess(train, test):

    lemmatizer = WordNetLemmatizer()

    replacements = get_replacements()

    

    train['ingredients'] = train['ingredients'].apply(lambda x: list(map(lambda y: y.lower(), x)))

    test['ingredients'] = test['ingredients'].apply(lambda x: list(map(lambda y: y.lower(), x)))

    

    stop_pattern = re.compile('[\d’%]')

    transform = lambda ingredients: tranform_to_single_string(ingredients, lemmatizer, replacements, stop_pattern)

    train['x'] = train['ingredients'].apply(transform)

    test['x'] = test['ingredients'].apply(transform)



    #show_unique_ingredients(train)

    

    vectorizer = make_pipeline(

        TfidfVectorizer(sublinear_tf=True),

        FunctionTransformer(lambda x: x.astype('float'), validate=False)

    )



    x_train = vectorizer.fit_transform(train['x'].values)

    x_train.sort_indices()

    x_test = vectorizer.transform(test['x'].values)

    return x_train, x_test
%%time

def main():

    train = pd.read_json('../input/train.json')

    test = pd.read_json('../input/test.json')

    

    train['num_ingredients'] = train['ingredients'].apply(lambda x: len(x))

    test['num_ingredients'] = test['ingredients'].apply(lambda x: len(x))

    train = train[train['num_ingredients'] > 2]

    

    x_train, x_test = preprocess(train, test)

    

    estimator = get_estimator()

    

    y_train = train['cuisine'].values

    classifier = OneVsRestClassifier(estimator, n_jobs=-1)

    classifier.fit(x_train, y_train)

    

    test['cuisine']  = classifier.predict(x_test)

    test[['id', 'cuisine']].to_csv('submission.csv', index=False)



main()