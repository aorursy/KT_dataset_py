import pandas as pd
import numpy as np
import itertools
import re
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
train = pd.read_json('../input/train.json')
train.head()
test = pd.read_json('../input/test.json')
cuisine = train['cuisine'].unique()
samples = train['cuisine'].value_counts()
ingredients = train['ingredients']
# Separate each type of cuisine and calculate the frequency each ingredient appears under each cuisine type
# Also Convert plural forms into singular as much as can
def list_str_plural(lst):
    str_cov = " ".join(lst)
    middle1 = re.sub(r'\d+', '', str_cov)     # Remove any digit from ingredients
    middle2 = middle1.replace('%','')
    middle3 = middle1.split(' ')
    middle4 = list(map(lambda word: WordNetLemmatizer().lemmatize(word.lower()),middle3))
    return ' '.join(middle4)

ingredients_new = ingredients.apply(list_str_plural)
train['ingredients'] = ingredients_new
train.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(cuisine)
le.transform(train['cuisine'])
tfidf = TfidfVectorizer(binary=True,use_idf=False)
X = tfidf.fit_transform(train['ingredients'])

train.head()
from sklearn.model_selection import GridSearchCV
param = {'C':[0.1,1,5, 10, 100],'gamma':[1,0.1,0.01,0.001,0.0001]}
from sklearn.svm import LinearSVC

lsvc = LinearSVC()
lsvc.fit(X,train['cuisine'])
# Test on test set
test['ingredients'] = test['ingredients'].apply(list_str_plural)
Xtest = tfidf.transform(test['ingredients'])
prediction = lsvc.predict(Xtest)
test['cuisine'] = prediction
new_test = pd.DataFrame(test[['id','cuisine']], columns=['id','cuisine'])
new_test.set_index('id')
prediction = new_test.to_csv('prediction.csv',index=False)