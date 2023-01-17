# importing the necessary modules

import nltk

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report, accuracy_score, log_loss

import warnings

import re

warnings.filterwarnings("ignore")

%matplotlib inline
# importing train data

mbti = pd.read_csv('../input/train.csv')
mbti.head()
mbti.info()
mbti['type'].value_counts().plot(kind='bar')

plt.show()
# M = 1 if extrovert M = 0 if introvert

mbti['mind'] = mbti['type'].apply(lambda x: 1 if x[0] == 'E' else 0)
# E = 1 if intuitive(N) E = 0 if observant(S)

mbti['energy'] = mbti['type'].apply(lambda x: 1 if x[1] == 'N' else 0)
# N = 1 if thinking N = 0 if feeling

mbti['nature'] = mbti['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
# T = 1 if judging N = 0 if prospecting

mbti['tactics'] = mbti['type'].apply(lambda x: 1 if x[3] == 'J' else 0)
mbti.head()
# mind category

labels = ['Extraversion', 'Introversion']

sizes = [mbti['mind'].value_counts()[1], mbti['mind'].value_counts()[0]]



fig, ax = plt.subplots(2, 2, figsize=(8, 8))

ax[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%',

             shadow=False, startangle=90)

ax[0, 0].axis('equal')



# energy category

labels = ['Intuitive', 'Observant']

sizes = [mbti['energy'].value_counts()[1], mbti['energy'].value_counts()[0]]



ax[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%',

             shadow=False, startangle=90)

ax[0, 1].axis('equal')



# nature category

labels = ['Thinking', 'Feeling']

sizes = [mbti['nature'].value_counts()[1], mbti['nature'].value_counts()[0]]



ax[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%',

             shadow=False, startangle=90)

ax[1, 0].axis('equal')



# tactics category

labels = ['Judging', 'Prospecting']

sizes = [mbti['tactics'].value_counts()[1], mbti['tactics'].value_counts()[0]]



ax[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%',

             shadow=False, startangle=90)

ax[1, 1].axis('equal')

plt.tight_layout()

plt.show()
# replacing url links with 'url-web'

first_pattern = '[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|'

second_pattern = '[!*,]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

pattern_url = r'http'+first_pattern+second_pattern



subs_url = r'url-web'

mbti['posts'] = mbti['posts'].replace(

                to_replace=pattern_url, value=subs_url, regex=True)
# replacing numbers with ''

pattern_numbers = r'\d+'

subs_numbers = r''

mbti['posts'] = mbti['posts'].replace(

                to_replace=pattern_numbers, value=subs_numbers, regex=True)
mbti['posts'] = mbti['posts'].str.lower()
# replacing '|||' with ' '

pattern_lines = r'\[|]|[|]|[|]+'

subs_lines = r' '

mbti['posts'] = mbti['posts'].replace(to_replace = pattern_lines, value = subs_lines, regex = True)
def remove_punctuation(post):

    return ''.join([l for l in post if l not in string.punctuation])
mbti['posts'] = mbti['posts'].apply(remove_punctuation)
mbti.head()
# building our vectorizer

count_vec = CountVectorizer(stop_words='english',

                            lowercase=True,

                            max_df=0.5,

                            min_df=2,

                            max_features=200)
# transforming posts to a matrix of token counts

X_count = count_vec.fit_transform(mbti['posts'])
X = X_count
y_mind = mbti['mind']
a, b, c, d = train_test_split(X, y_mind, test_size=0.2, random_state=42)

X_train_mind = a

X_test_mind = b

y_train_mind = c

y_test_mind = d
svm = SVC()

params = {'C': [1.0, 0.001, 0.01],

          'gamma': ['auto', 'scale', 1.0, 0.001, 0.01]}

clf_mind = GridSearchCV(estimator=svm, param_grid=params)

clf_mind.fit(X_train_mind, y_train_mind)
#  gives parameter setting that returned the best results

clf_mind.best_params_
y_pred_mind = clf_mind.predict(X_test_mind)

print("Accuracy score:", accuracy_score(y_test_mind, y_pred_mind))
y_energy = mbti['energy']
a, b, c, d = train_test_split(X, y_energy, test_size=0.2, random_state=42)

X_train_energy = a

X_test_energy = b

y_train_energy = c

y_test_energy = d
svm = SVC()

params = {'C': [1.0, 0.001, 0.01],

          'gamma': ['auto', 'scale', 1.0, 0.001, 0.01]}

clf_energy = GridSearchCV(estimator=svm, param_grid=params)

clf_energy.fit(X_train_energy, y_train_energy)
# gives parameter setting that returned the best results

clf_energy.best_params_
y_pred_energy = clf_energy.predict(X_test_energy)

print("Accuracy score:", accuracy_score(y_test_energy, y_pred_energy))
y_nature = mbti['nature']
a, b, c, d = train_test_split(X, y_nature, test_size=0.2, random_state=42)

X_train_nature = a

X_test_nature = b

y_train_nature = c

y_test_nature = d
svm = SVC()

params = {'C': [1.0, 0.001, 0.01],

          'gamma': ['auto', 'scale', 1.0, 0.001, 0.01]}

clf_nature = GridSearchCV(estimator=svm, param_grid=params)

clf_nature.fit(X_train_nature, y_train_nature)
# gives parameter setting that returned the best results

clf_nature.best_params_
y_pred_nature = clf_nature.predict(X_test_nature)

print("Accuracy score:", accuracy_score(y_test_nature, y_pred_nature))
y_tactics = mbti['tactics']
a, b, c, d = train_test_split(X, y_tactics, test_size=0.2, random_state=42)

X_train_tactics = a

X_test_tactics = b

y_train_tactics = c

y_test_tactics = d
svm = SVC()

params = {'C': [1.0, 0.001, 0.01, 0.1],

          'gamma': ['auto', 'scale', 1.0, 0.001, 0.01, 0.1]}

clf_tactics = GridSearchCV(estimator=svm, param_grid=params)

clf_tactics.fit(X_train_tactics, y_train_tactics)
# gives parameter setting that returned the best results

clf_tactics.best_params_
y_pred_tactics = clf_tactics.predict(X_test_tactics)

print("Accuracy score:", accuracy_score(y_test_tactics, y_pred_tactics))
# importing test data

test_mbti = pd.read_csv('../input/test.csv')
test_mbti.head()
# replacing url links with 'url-web'

x = pattern_url

y = subs_url

z = True

test_mbti['posts'] = test_mbti['posts'].replace(to_replace=x, value=y, regex=z)
# replacing numbers with ''

a = pattern_numbers

b = subs_numbers

c = True

test_mbti['posts'] = test_mbti['posts'].replace(to_replace=a, value=b, regex=c)
test_mbti['posts'] = test_mbti['posts'].str.lower()
test_mbti['posts'] = test_mbti['posts'].replace(to_replace = pattern_lines, value = subs_lines, regex = True)
test_mbti['posts'] = test_mbti['posts'].apply(remove_punctuation)
test_mbti.head()
X_count_test = count_vec.fit_transform(test_mbti['posts'])

X_test = X_count_test
# mind predictions of the test data

mind_predictions = clf_mind.predict(X_test)
# energy predictions of the test data

energy_predictions = clf_energy.predict(X_test)
# nature predictions of the test data

nature_predictions = clf_nature.predict(X_test)
# tactics predictions of the test data

tactics_predictions = clf_tactics.predict(X_test)
# creating a dataframe to be exported to a csv file,

# containing all the relevant column names.

listym = ['id', 'mind', 'energy', 'nature', 'tactics']

submission = pd.DataFrame(columns=listym)
submission['id'] = test_mbti['id']
submission['mind'] = mind_predictions



submission['energy'] = energy_predictions



submission['nature'] = nature_predictions



submission['tactics'] = tactics_predictions
# saving submission to csv file

submission.to_csv('submission.csv', index=False)