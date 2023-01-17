# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

sarc_df = pd.read_json("../input/Sarcasm_Headlines_Dataset.json", lines=True, orient='columns', date_unit ='s', dtype=False)

# Any results you write to the current directory are saved as output.
# Extract just the domain name from the url in article_link column

sarc_df['article_link'].replace({'.*://\w+\.(\w+)\..*' : '\\1'}, inplace=True, regex=True)
sarc_df.dropna(1, how='all')

sarc_df['corpus']=sarc_df['article_link'] + ' ' + sarc_df['headline']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(sarc_df['corpus'], 

                                                    sarc_df['is_sarcastic'], 

                                                    random_state=0)



from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=5, stop_words = 'english',analyzer='word', ngram_range=(1,2)).fit(X_train)

print('Vocabulary len:', len(vect.get_feature_names()))

print('Longest word:', max(vect.vocabulary_, key=len))



X_train_vectorized = vect.transform(X_train)
#from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

lsvcmodel = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='warn',

          n_jobs=None, penalty='l2', random_state=None, solver='warn',

          tol=0.0001, verbose=0, warm_start=False)

X_train_vectorized = vect.transform(X_train)

lsvcmodel.fit(X_train_vectorized, y_train)



from sklearn.metrics import accuracy_score



y_predsvc = lsvcmodel.predict(vect.transform(X_test))

print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_predsvc) * 100))