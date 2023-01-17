# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('darkgrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)

df.dropna(axis=0, how='any', inplace=True)

df.head()
print(df.shape)

df.describe()
# have to use ints for classification

df['average_rating_bin'] = pd.cut(df['average_rating'], 

                                  np.linspace(0, 5, 11), 

                                  labels=np.linspace(0, 9, 10)).fillna(0).astype(int)

df['# num_pages_bin'] = pd.cut(df['# num_pages'], 

                                  np.linspace(0, 7000, 29), 

                                  labels=np.linspace(0, 6750, 28)).fillna(0).astype(int)

df['average_rating_bin'].value_counts(), df['# num_pages_bin'].value_counts()
# average_ratingclustered between 3-5 stars, lower ratings will being noise.

# To make accurate predictions, limit data to ratings with 3-5 stars.



df = df[df['average_rating'] >= 3]



# have to use ints for classification

df['average_rating_bin'] = pd.cut(df['average_rating'], 

                                  np.linspace(3, 5, 11), 

                                  labels=np.linspace(0, 9, 10)).fillna(0).astype(int)
'''

average rating is as follows:

0 - [3.0, 3.2]

1 - [3.2, 3.4]

2 - [3.4, 3.6]

...

9 - [4.8, 5.0]

'''

sns.pairplot(df, hue='average_rating_bin')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LassoCV



# ignore ID variables, titles and authors, and languages

X_cols = [x for x in df.columns if x not in ['bookID', 'title', 'authors', 

                                             'isbn', 'isbn13', 'language_code',

                                            'average_rating', 'average_rating_bin',

                                            '# num_pages_bin']]

print(X_cols)

df.dropna(subset=X_cols, axis=0, how='any', inplace=True)

X = df[X_cols]

y = df['average_rating_bin']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



clf = LassoCV().fit(X_train, y_train)



clf.alpha_, clf.coef_
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



params = {'max_depth': range(1, 5), 

          'max_features': range(1, 4), 

          'min_samples_leaf': np.arange(5, 26, 5)}



clf = GridSearchCV(DecisionTreeClassifier(), 

                   params,

                   iid=False,

                   n_jobs=1,

                   cv=5)

clf.fit(X_train, y_train)

tree_model = clf.best_estimator_



y_pred = tree_model.predict(X_test)



print(clf.best_score_, clf.best_params_, end='\n\n') # mean CV score

print(classification_report(y_test, y_pred)) # only for classification

print(confusion_matrix(y_test, y_pred)) # only for classification
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



rbf_svc = SVC(kernel='rbf', decision_function_shape='ovo').fit(X_train, y_train)



y_pred = rbf_svc.predict(X_test)



print(classification_report(y_test, y_pred)) # only for classification

print(confusion_matrix(y_test, y_pred)) # only for classification

print(accuracy_score(y_test, y_pred)) # only for classification