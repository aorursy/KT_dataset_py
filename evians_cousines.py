import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from collections import Counter
pd.options.display.max_colwidth = 110
df = pd.read_json('../input/train.json').set_index('id')
df_test = pd.read_json('../input/test.json').set_index('id')
df.head(10)
df['cuisine'].unique()
df['cuisine'].value_counts()
df[df['ingredients'].str.len() < 2]
sns.countplot(y='cuisine', order=df['cuisine'].value_counts().reset_index()['index'], data=df)
plt.title("Cuisines")
sns.boxplot(x='cuisine', y=df['ingredients'].str.len(), data=df)
plt.gcf().set_size_inches(22, 10)
plt.title('Number of ingredients')
# Most common ingredients
ingredients = [item for sublist in df['ingredients'] for item in sublist]
counter = Counter(ingredients)
top_ingredients = counter.most_common(15)
df_ingredients = pd.DataFrame(top_ingredients, columns=['ingredient', 'count'])
sns.barplot(y='ingredient', x='count', data=df_ingredients)
plt.title('Most common ingredients')
# df_italian = df[df.cuisine == 'italian']
# ingredients = [item for sublist in df_italian['ingredients'] for item in sublist]
# counter = Counter(ingredients)
# top_ingredients = counter.most_common(10)
# df_ingredients = pd.DataFrame(top_ingredients, columns=['ingredient', 'count'])
# sns.barplot(y='ingredient', x='count', data=df_ingredients)
# plt.title('Italian ingredients')
# df_italian = df[df.cuisine == 'french']
# ingredients = [item for sublist in df_italian['ingredients'] for item in sublist]
# counter = Counter(ingredients)
# top_ingredients = counter.most_common(10)
# df_ingredients = pd.DataFrame(top_ingredients, columns=['ingredient', 'count'])
# sns.barplot(y='ingredient', x='count', data=df_ingredients)
# plt.title('French ingredients')
# df_italian = df[df.cuisine == 'chinese']
# ingredients = [item for sublist in df_italian['ingredients'] for item in sublist]
# counter = Counter(ingredients)
# top_ingredients = counter.most_common(10)
# df_ingredients = pd.DataFrame(top_ingredients, columns=['ingredient', 'count'])
# sns.barplot(y='ingredient', x='count', data=df_ingredients)
# plt.title('Chinese ingredients')
df = df.drop(df[df['ingredients'].str.len() < 2].index, axis=0)
df = df.drop(df[df['ingredients'].str.len() > 30].index, axis=0)
dfX = df['ingredients'].str.join(' ').str.lower()
dfX_test = df_test['ingredients'].str.join(' ').str.lower()
dfy = df['cuisine']
lbe = LabelEncoder()
y = lbe.fit_transform(dfy.values)
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(dfX.values)
X_test = tfidf.transform(dfX_test.values)
clf = SVC(C=100, # penalty parameter
          kernel='rbf', # kernel type, rbf working fine here
          degree=3, # default value
          gamma=1, # kernel coefficient
          coef0=1, # change to 1 from default value of 0.0
          shrinking=True, # using shrinking heuristics
          tol=0.001, # stopping criterion tolerance 
          probability=False, # no need to enable probability estimates
          cache_size=200, # 200 MB cache size
          class_weight=None, # all classes are treated equally 
          verbose=False, # print the logs 
          max_iter=-1, # no limit, let it run
          decision_function_shape=None, # will use one vs rest explicitly 
          random_state=None)

# model = OneVsRestClassifier(clf)
# model.fit(X, y)
clf.fit(X, y)
y_test = clf.predict(X_test)
y_pred = lbe.inverse_transform(y_test)
cross_val_score(clf, X, y, cv=3)
sub = pd.DataFrame({'id':df_test.index, 'cuisine':y_pred})
sub.to_csv('cuisine_output.csv', index=False)
