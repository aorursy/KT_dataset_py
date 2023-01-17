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
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# train = train.drop(['keyword', 'location'], axis=1)

test = test.drop(['keyword', 'location'], axis=1)
test.head()
X = train['text']

y = train['target']

X_test = test['text']
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer





text_clf = Pipeline([('tfidf', TfidfVectorizer()),

                    ('model', LinearSVC())])



text_clf.fit(X, y)

prediction = text_clf.predict(X_test)
prediction
submission = pd.DataFrame({

        "Id": test["id"],

        "target": prediction

    })

submission.to_csv('submission.csv', index=False)