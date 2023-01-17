# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df_train = pd.read_csv('/kaggle/input/creditscoringds/df_train.csv', sep=',')

df_test = pd.read_csv('/kaggle/input/creditscoringds/df_test.csv', sep=',')

df_test = df_test.drop(['ProductCategory_ticket'], axis=1)



tr_id = pd.read_csv('/kaggle/input/mydata/Test.csv', sep=',')

tr_id.head()
df_test.head()
y_test = tr_id[['TransactionId']]

y_test['IsDefaulted'] = 1

y_test.head()
X_test = df_test

X_test.head()
df_train.head()
df_train_X = df_train.drop('IsDefaulted', axis=1)

df_train_X.head()
df_train_y = df_train[['IsDefaulted']]

df_train_y.head()
from sklearn.model_selection import train_test_split



X_all = df_train_X.values

y_all = df_train_y.values

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 

                                                     test_size=0.3, 

                                                     random_state=17)
'''from sklearn.model_selection import train_test_split

X_all = df_train.drop('IsDefaulted', axis=1)

y_all = df_train['IsDefaulted']

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 

                                                     test_size=0.3, 

                                                     random_state=17)'''
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier





logistic = LogisticRegression(random_state=17,

                              penalty='elasticnet',

                              solver='saga',

                              l1_ratio=0.5,

                              class_weight={0:0.4, 1:0.6})

support = LinearSVC(random_state=17)

neighbors = KNeighborsClassifier(n_neighbors=5)

'''vote = VotingClassifier(estimators=[('lr', logistic), ('knn', neighbors),

                                    ('svc', support)])'''

vote = VotingClassifier(estimators=[('lr', logistic),

                                    ('svc', support)])



logistic.fit(X_train, y_train)

support.fit(X_train, y_train)

neighbors.fit(X_train, y_train)

vote.fit(X_train, y_train)

print(vote.score(X_valid, y_valid))

y_pred = vote.predict(X_valid)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_valid, y_pred)
vote.fit(X_all, y_all)
#y_test['IfDefaulted'] = logistic.predict(df_test)

#y_test.head()

y_x = vote.predict(df_test)

for i in range(y_test.shape[0]):

    y_test.at[i,'IsDefaulted'] = y_x[i]

y_test.head()
y_test.to_csv('ANSWER_vote.csv',index=False)