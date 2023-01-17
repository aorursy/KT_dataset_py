# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# importing necessary libraries

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.svm import SVC

import json



import matplotlib.pyplot as plt

%matplotlib inline



%config InlineBackend.figure_format = 'retina'
# Reading the datasets

train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')



# let's lokk at few rows in the training set

train.head()
# let's look at the training data details

train.info()
# how about the test data

test.info()
# how many cuisine are present in training data

print("Total number of cuisines are: {}".format(len(train['cuisine'].unique())))

print(train['cuisine'].unique())
# let's see which cuisine is very popular among users by counting the frequency of the cusines

cuisine_df = pd.DataFrame({

                          'Count': train['cuisine'].value_counts()})

cuisine_df
fig, ax = plt.subplots(figsize=(16, 9))

cuisine_df.plot(kind='bar', ax=ax)



plt.show()
# list out all ingredients

allingredients = []

for item in train['ingredients']:

    for ingr in item:

        allingredients.append(ingr)
from collections import Counter

count_ingr = Counter(allingredients)

# for ingr in allingredients:

#     count_ingr[ingr]

count_ingr
# top 10 most used ingredients

count_ingr.most_common(20)
count_ingr = dict(count_ingr)

ingr_count = pd.Series(count_ingr)

fig, ax1 = plt.subplots(figsize=(16, 9))

ingr_count.sort_values(ascending=False)[:10].plot(kind='bar', ax=ax1)

plt.show()
train['num_ingredients'] = train['ingredients'].apply(len)

train = train[train['num_ingredients'] > 1]

train.drop('num_ingredients', axis=1, inplace=True)
def case_normalization(ingredients):

    ingredients = ' '.join(ingredients).lower()

    ingredients = ingredients.replace('-','')

    return ingredients
train['ingredients'] = train['ingredients'].apply(case_normalization)

test['ingredients']  = test['ingredients'].apply(case_normalization)

train.head()
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import re

def clean_data(ingredients):

    lemmatizer = WordNetLemmatizer()

    ingredients = re.sub(r'[^\w\s\d]', '', ingredients)

    ingredients = re.sub(r'\d+', '', ingredients)

    tokens = word_tokenize(ingredients)

    words = []

    for token in tokens:

        word = lemmatizer.lemmatize(token)

        if len(word) > 0:

            words.append(word)

    return ' '.join(words)

    
train['features'] = train['ingredients'].apply(clean_data)

test['features'] = test['ingredients'].apply(clean_data)

train.head()
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import FunctionTransformer, LabelEncoder

pipeline = Pipeline([

    ("tfidf", TfidfVectorizer()),

    ("functiontransform", FunctionTransformer(lambda x: x.astype('float32'), validate=False))

])



x_train = pipeline.fit_transform(train['features'].values)

x_train.sort_indices()

x_test = pipeline.transform(test['features'].values)
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train['cuisine'].values)

dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

estimator = SVC(

    C = 50,

    kernel = 'rbf',

    gamma = 1.5,

    random_state=42

)

classifier = OneVsRestClassifier(estimator, n_jobs=-1)
classifier.fit(x_train, y_train)
y_pred = label_encoder.inverse_transform(classifier.predict(x_test))

test['cuisine'] = y_pred

test[['id', 'cuisine']].to_csv('submission.csv', index=False)

test[['id', 'cuisine']].head()
!ls