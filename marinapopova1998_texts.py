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
from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(

                    subset='all', 

                    categories=['alt.atheism', 'sci.space']

             )
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
tmp.loc[2][0]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["data"])
grid = {'C': np.power(10.0, np.arange(-5, 6))}

cv = KFold(n_splits=5, shuffle=True, random_state=241)

clf = SVC(kernel='linear', random_state=241)

gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

gs.fit(X, newsgroups.target)
lr = LogisticRegression()

lr.fit(X,data["target"])
gs.best_score_
gs.best_params_
clf = SVC(kernel='linear', random_state=241, C=1.0)
clf.fit(X, newsgroups.target)
idx = np.argsort(np.abs(clf.coef_.toarray()[0]))[-10:]
words = np.array(vectorizer.get_feature_names())[idx]
" ".join(sorted(words))