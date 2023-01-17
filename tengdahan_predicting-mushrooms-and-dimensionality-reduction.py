import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
shroom = pd.read_csv('../input/mushrooms.csv')

shroom[:3]
for feature in shroom.columns:

    print(feature, ':', shroom[feature].unique())
shroom = shroom.drop(shroom[shroom['stalk-root']=='?'].index)

shroom = shroom.drop('veil-type', axis=1)
# making sure no other missing values

shroom.isnull().sum().sum()
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

for feature in shroom.columns:

    if len(shroom[feature].unique()) == 2:

        shroom[feature] = lb.fit_transform(shroom[feature])
features_onehot = []

for feature in shroom.columns[1:]:

    if len(shroom[feature].unique()) > 2:

        features_onehot.append(feature)

temp = pd.get_dummies(shroom[features_onehot])

shroom = shroom.join(temp)

shroom = shroom.drop(features_onehot, axis=1)

shroom[:3]
# making sure all values are numerical

np.unique(shroom.values)
from sklearn.model_selection import train_test_split



X = shroom.drop('class', axis=1).values

y = shroom['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

print('X_train Shape:', X_train.shape)

print('X_test Shape:', X_test.shape)

print('y_train Shape:', y_train.shape)

print('y_test Shape:', y_test.shape)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)

print('Training Accuracy: %.2f%%' % (lr.score(X_train, y_train)*100))

print('Test Accuracy: %.2f%%' % (lr.score(X_test, y_test)*100))
mushroom = pd.read_csv('../input/mushrooms.csv')

shroom2 = mushroom.drop(mushroom[mushroom['stalk-root']=='?'].index)
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
lbe = LabelEncoder()

for feature in shroom2.columns[1:]:

    shroom2[feature] = lbe.fit_transform(shroom2[feature])

shroom2[:3]
y = shroom2['class'].values

X = shroom2.drop('class', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)

rfc.fit(X_train, y_train)

print('Training Score: %.2f%%' % (rfc.score(X_train, y_train) * 100))

print('Test Score: %.2f%%' % (rfc.score(X_test, y_test) * 100))
importances = rfc.feature_importances_

features = shroom2.columns[1:]

sort_indices = np.argsort(importances)[::-1]

sorted_features = []

for idx in sort_indices:

    sorted_features.append(features[idx])

plt.figure()

plt.bar(range(len(importances)), importances[sort_indices], align='center');

plt.xticks(range(len(importances)), sorted_features, rotation='vertical');

plt.xlim([-1, len(importances)])

plt.grid(False)
# random forest

top_features = sorted_features[:2]

y = shroom2['class'].values

X = shroom2[top_features].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)

rfc.fit(X_train, y_train)

print('Random Forest with 2 features')

print('Training Score: %.2f%%' % (rfc.score(X_train, y_train) * 100))

print('Test Score: %.2f%%' % (rfc.score(X_test, y_test) * 100))
# logistic regression

top_features_2 = [

    'odor_a', 'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n', 'odor_p', # feature 1

    'spore-print-color_h', 'spore-print-color_k','spore-print-color_n', # feature 2

    'spore-print-color_r', 'spore-print-color_u', 'spore-print-color_w',

]

y = shroom['class'].values

X = shroom[top_features_2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

lr = LogisticRegression()

lr.fit(X_train, y_train)

print('Logistic Regression with 2 features')

print('Training Accuracy: %.2f%%' % (lr.score(X_train, y_train)*100))

print('Test Accuracy: %.2f%%' % (lr.score(X_test, y_test)*100))
# random forest

top_features = sorted_features[:1]

y = shroom2['class'].values

X = shroom2[top_features].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)

rfc.fit(X_train, y_train)

print('Random Forest with 1 feature')

print('Training Score: %.2f%%' % (rfc.score(X_train, y_train) * 100))

print('Test Score: %.2f%%' % (rfc.score(X_test, y_test) * 100))
# logistic regression

top_features_2 = [

    'odor_a', 'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n', 'odor_p', # feature 1

]

y = shroom['class'].values

X = shroom[top_features_2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

lr = LogisticRegression()

lr.fit(X_train, y_train)

print('Logistic Regression with 1 feature')

print('Training Accuracy: %.2f%%' % (lr.score(X_train, y_train)*100))

print('Test Accuracy: %.2f%%' % (lr.score(X_test, y_test)*100))
sns.factorplot('class', col='odor', data=mushroom, kind='count', size=2.5, aspect=0.6, col_wrap=5);
g = sns.factorplot('class', col='spore-print-color', data=mushroom,

               kind='count', size=2.5, aspect=0.6, col_wrap=5)

g.set_titles("{col_name}")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('spore-print-color');
sns.factorplot('class', col='gill-attachment', data=mushroom, kind='count', size=2.5, aspect=0.6, col_wrap=5);