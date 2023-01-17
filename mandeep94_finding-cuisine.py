# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import re

from nltk.stem import WordNetLemmatizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_json('../input/train.json')

train.head()
train['ingredient_list'] = [','.join(z).strip() for z in train['ingredients']]
train.head()
test = pd.read_json('../input/test.json')

test.head()
ingredients = train['ingredient_list']

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(ingredients).todense()
cuisines = train['cuisine']

cuisines.head()
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(tfidf_matrix, cuisines)
test['ingredient_list'] = [','.join(z).strip() for z in test['ingredients']]

test.head()

test_ingredients = test['ingredient_list']

test_tfidf_matrix = vectorizer.transform(test_ingredients)

test_cuisines = clf.predict(test_tfidf_matrix)
test['cuisine'] = test_cuisines
test.head()
test[['id' , 'cuisine' ]].to_csv("submission.csv", index=False)
test[['id','cuisine']].head()