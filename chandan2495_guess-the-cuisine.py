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
import re
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')
train_df.head()
print('Training size : {}\nTest size : {}'.format(train_df.shape[0],test_df.shape[0]))
print('Number of unique cuisines ' + str(len(train_df['cuisine'].value_counts())))
import matplotlib.pyplot as plt
import seaborn as sns
cuisine_name = train_df['cuisine'].value_counts().index
cuisine_count = train_df['cuisine'].value_counts().values
plt.figure(figsize=(20,5))
sns.barplot(cuisine_name, cuisine_count)
plt.xlabel('Cuisine')
plt.ylabel('#Occurrences')
plt.title('Most common cuisines bar plot.')
plt.show()
# bar plot of most common ingredients
ingredients = list()
for ings in train_df['ingredients']:
    for ing in ings:
        ingredients.append(ing)
print('Total number of ingredients (including duplicates): ' + str(len(ingredients)))
from collections import Counter
ingredients_counter = Counter(ingredients)
k = 15
plt.figure(figsize=(20,5))
top_k_ingredients = ingredients_counter.most_common(k)
name, counts = zip(*top_k_ingredients)
sns.barplot(list(name), list(counts))
plt.show()
train_df['ingredients'] = train_df['ingredients'].map(lambda x: " ".join(x))
test_df['ingredients'] = test_df['ingredients'].map(lambda x: " ".join(x))
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf = TfidfVectorizer()
train_features = tfidf.fit_transform(train_df['ingredients'])
test_features = tfidf.transform(test_df['ingredients'])
count_vectorizer = CountVectorizer()
train_features_cv = count_vectorizer.fit_transform(train_df['ingredients'])
test_features_cv = count_vectorizer.transform(test_df['ingredients'])
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['cuisine'])
X = pd.DataFrame(train_features.todense())
X_cv = pd.DataFrame(train_features_cv.todense())
X_tfidf_cv = pd.concat([X,X_cv],axis=1)
y = train_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X_train_cv, X_test_cv, y_train, y_test = train_test_split(X_cv, y, random_state=42, test_size=0.2)
X_train_tfidf_cv, X_test_tfidf_cv, y_train, y_test = train_test_split(X_cv, y, random_state=42, test_size=0.2)
# TF-IDF features
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
# Count Vectorizer features
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=30)
# rfc.fit(X_train_cv, y_train)
# rfc.score(X_test_cv, y_test)
# Count Vectorizer and TF-IDF combined features
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=30)
# rfc.fit(X_train_tfidf_cv, y_train)
# rfc.score(X_test_tfidf_cv, y_test)
test_X = pd.DataFrame(test_features.todense())
test_X_cv = pd.concat([pd.DataFrame(test_features.todense()),pd.DataFrame(test_features_cv.todense())],axis=1)
predict_test = rfc.predict(test_X)
predicted_cuisines = label_encoder.inverse_transform(predict_test)
pd.DataFrame({'id':test_df['id'], 'cuisine':predicted_cuisines}).to_csv('submission.csv', index=False)
predicted_cuisines[:10]
predicted = rfc.predict(X_train)
np.mean(y_train==predicted)
len(predicted_cuisines)
