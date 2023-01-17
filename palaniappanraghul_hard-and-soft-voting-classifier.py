# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

from sklearn.datasets import make_classification, make_regression

from sklearn.model_selection import train_test_split

                                     

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler



from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier



from collections import Counter
X, y = make_classification(n_samples=500, 

                           n_features=10,

                           random_state=42)



X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.2, 

                                                    stratify=y, 

                                                    random_state=42)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


clf_list = [('decision tree', DecisionTreeClassifier()),

            ('logistic regression', LogisticRegression()),

            ('knn', KNeighborsClassifier()),

            ('naive bayes classifier', GaussianNB())]
for model_tuple in clf_list:

    model = model_tuple[1]

    if 'random_state' in model.get_params().keys():

        model.set_params(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_pred, y_test)

    print(f"{model_tuple[0]}'s accuracy: {acc:.2f}")
voting_clf = VotingClassifier(clf_list, voting='hard')

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)

print(f"Voting Classifier's accuracy: {accuracy_score(y_pred, y_test):.2f}")
voting_clf = VotingClassifier(clf_list, voting='soft')

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)

print(f"Voting Classifier's accuracy: {accuracy_score(y_pred, y_test):.2f}")