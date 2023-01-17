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

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.calibration import CalibratedClassifierCV



models = [('MultiNB', MultinomialNB(alpha=0.03)),

          ('BernoulliNB', BernoulliNB(alpha=0.03)),

          ('Huber', SGDClassifier(loss='modified_huber')),

          ('LR', LogisticRegression(C=30))]



df_train = pd.read_csv('/kaggle/input/spooky-author-identification/train.zip')

df_test = pd.read_csv('/kaggle/input/spooky-author-identification/test.zip')

df_test.head()
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))



clf = VotingClassifier(models, voting='soft')

X_train = vectorizer.fit_transform(df_train['text'].values)

authors = ['MWS','EAP','HPL']

y_train = df_train['author'].apply(authors.index).values

clf.fit(X_train, y_train)
X_test = vectorizer.transform(df_test.text.values)

results = clf.predict_proba(X_test)

pd.DataFrame(results, index=df_test.id, columns=authors).to_csv('sub4.csv')