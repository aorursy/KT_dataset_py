# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import re

from nltk.stem import WordNetLemmatizer



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import FunctionTransformer



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')



class_names = list(train.cuisine.unique())

class_names
train['num_ings'] = train['ingredients'].apply(lambda x : len(x))

test['num_ings'] = test['ingredients'].apply(lambda x : len(x))

train.head()
len(train)
train = train[train['num_ings'] > 2]

len(train)
train['ingredients'] = train['ingredients'].apply(lambda x: list(map(lambda y: y.lower(), x)))

test['ingredients'] = test['ingredients'].apply(lambda x: list(map(lambda y: y.lower(), x)))

train.head()
def get_replacements():

    return {'wasabe': 'wasabi', '-': '', 'sauc': 'sauce',

            'baby spinach': 'babyspinach', 'coconut cream': 'coconutcream',

            'coriander seeds': 'corianderseeds', 'corn tortillas': 'corntortillas',

            'cream cheese': 'creamcheese', 'fish sauce': 'fishsauce',

            'purple onion': 'purpleonion','refried beans': 'refriedbeans', 

            'rice cakes': 'ricecakes', 'rice syrup': 'ricesyrup', 

            'sour cream': 'sourcream', 'toasted sesame seeds': 'toastedsesameseeds', 

            'toasted sesame oil': 'toastedsesameoil', 'yellow onion': 'yellowonion'}
lemmatizer = WordNetLemmatizer()

replacements = get_replacements()

stop_pattern = re.compile('[\d’%]')
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
transform = lambda ingredients: tranform_to_single_string(ingredients, lemmatizer, replacements, stop_pattern)

train['x'] = train['ingredients'].apply(transform)

test['x'] = test['ingredients'].apply(transform)

train.head()
vectorizer = make_pipeline(

        TfidfVectorizer(sublinear_tf=True),

        FunctionTransformer(lambda x: x.astype('float'), validate=False)

    )
x_train = vectorizer.fit_transform(train['x'].values)

x_train.sort_indices()

x_test = vectorizer.transform(test['x'].values)
print(x_train[0])
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
estimator = get_estimator()

y_train = train['cuisine'].values

classifier = OneVsRestClassifier(estimator, n_jobs=-1)

classifier.fit(x_train, y_train)
test['cuisine']  = classifier.predict(x_test)

test[['id', 'cuisine']].to_csv('submission.csv', index=False)