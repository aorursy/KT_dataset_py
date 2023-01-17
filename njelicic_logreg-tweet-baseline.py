# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# imports if fails run '!pip3 install [package name]'

from sklearn.metrics import f1_score



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_train










tfidf_vectorizer = TfidfVectorizer(ngram_range=(3,3),stop_words='english',analyzer='char_wb')





X = tfidf_vectorizer.fit_transform(df_train['text'].values)



y = df_train['target'].values



models = []

n_splits = 3

fold = 0 

fs = []

for train_index, test_index in StratifiedKFold(n_splits=n_splits).split(X, y):



    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]



    clf = LogisticRegression(C=2,solver='lbfgs')



    clf.fit(X_train,y_train)

    preds = clf.predict(X_test)

    f1 = f1_score(y_test,preds)

    fs.append(f1)

    models.append(clf)

    fold += 1











print(np.mean(fs))
X_test = tfidf_vectorizer.transform(df_test['text'].values)



y_hat = clf.predict(X_test)
final = np.zeros((X_test.shape[0]))



for i in range(n_splits):

        clf = models[i]

        preds = clf.predict(X_test)

        

        final += preds/n_splits



    

final = np.where(final>=0.5,1,0)
import seaborn as sns



sns.distplot(final)
submission = df_train = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv') 
submission['target'] = final
submission.to_csv('submission.csv',index=False)