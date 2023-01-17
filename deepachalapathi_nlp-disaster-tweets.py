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
train = pd.read_csv('../input/nlp-getting-started/train.csv')

print(train.info())
train.isnull().sum()
#we don't need to fill in the null values of these columns as we don't use these columns
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(binary = True, max_features = 1500)
#X = cv.fit_transform(train['text']).toarray()
#y = train.iloc[:, -1].values
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(max_features=2000,binary = True)
X = tf_idf.fit_transform(train['text']).toarray()
y = train.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5, scoring="f1")
print(scores)
test = pd.read_csv('../input/nlp-getting-started/test.csv')
X_test = tf_idf.transform(test['text']).toarray()
X_test = sc.transform(X_test)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission['target'] = classifier.predict(X_test)
sample_submission.head(10)
sample_submission.to_csv('submission.csv', index=False)