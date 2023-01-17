# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')

df.head()
# df.info()
df.churn.value_counts()
sns.countplot(df['churn'])
# df = df.join(pd.get_dummies(df['state']))

# После создания дополнительных признаков на основе столбца 'state' accuracy понижается.
states = df.state.unique()

states_num = {}

for i in range(len(states)):

    states_num[states[i]] = i

df['states_num'] = df['state'].map(states_num)
df.drop('state', axis=1, inplace=True)



df['churn_'] = df['churn'].map({True: 1, False: 0})

df.drop('churn', axis=1, inplace=True)



df['international_plan'] = df['international plan'].map({'yes': 1, 'no': 0})

df.drop('international plan', axis=1, inplace=True)



df['voice_mail_plan'] = df['voice mail plan'].map({'yes': 1, 'no': 0})

df.drop('voice mail plan', axis=1, inplace=True)



df.drop('phone number', axis=1, inplace=True)
# df.info()
y = df['churn_']

X = df.drop(labels = ['churn_'],axis = 1) 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

scaler.mean_

X_new = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_new, 

                                                      y, 

                                                      test_size=0.3, 

                                                      random_state=19)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)
from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X_new, y, 

                         cv=kf, scoring='accuracy')

mean_score = scores.mean()

mean_score
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=kf)

knn_grid.fit(X_train, y_train)
knn_grid.best_score_
knn_grid.best_params_
score_df = pd.DataFrame(knn_grid.cv_results_)

# score_df.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

plt.plot(score_df['param_n_neighbors'], score_df['mean_test_score'])
p = np.linspace(1,10,200)

p_dict ={}

max_acc = 0

for val in p:

    knn = KNeighborsClassifier(n_neighbors=5, p=val, metric='minkowski', weights='distance')

    scores = cross_val_score(knn, X_new, y, 

                         cv=kf, scoring='accuracy')

    mean_score = scores.mean()

    if mean_score > max_acc:

        max_acc = mean_score

        max_acc_str = 'p = {}:   accuracy = {}'.format(val, max_acc)

    p_dict[val] = mean_score

    

max_acc_str
from sklearn.neighbors import NearestCentroid

clf = NearestCentroid()

clf.fit(X_train, y_train)

scores = cross_val_score(knn, X_new, y, 

                         cv=kf, scoring='accuracy')

mean_score = scores.mean()

mean_score
from sklearn.neighbors import RadiusNeighborsClassifier

neigh = RadiusNeighborsClassifier(radius=1.0)

neigh.fit(X_train, y_train)

scores = cross_val_score(knn, X_new, y, 

                         cv=kf, scoring='accuracy')

mean_score = scores.mean()

mean_score