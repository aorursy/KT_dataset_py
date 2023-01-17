# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib notebook

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

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
spam = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding = 'latin-1')

spam.head(10)
spam = spam.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

spam['v1'] = np.where(spam['v1'] == 'spam',1,0)

spam.head()
spam.isnull().sum()
plt.figure()

sns.countplot(spam['v1'])

plt.plot
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam['v2'], spam['v1'], random_state = 0 )
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train)

X_test_cv = cv.transform(X_test)
from sklearn.naive_bayes import MultinomialNB

clf_multi = MultinomialNB(alpha = 0.1).fit(X_train_cv, y_train)

predictions = clf_multi.predict(X_test_cv)

acc_multi = accuracy_score(y_test, predictions)

acc_multi
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X_train_cv, y_train)

predictions = model.predict(X_test_cv)

acc_lr = accuracy_score(y_test, predictions)

acc_lr
from sklearn.svm import SVC

svc = SVC(kernel = 'linear').fit(X_train_cv, y_train)

predictions = svc.predict(X_test_cv)

acc_svc = accuracy_score(y_test, predictions)

acc_svc
from sklearn.ensemble import RandomForestClassifier

clf_rfc = RandomForestClassifier(random_state = 0).fit(X_train_cv, y_train)

predictions = clf_rfc.predict(X_test_cv)

acc_rfc = accuracy_score(y_test, predictions)

acc_rfc
Models = ({

    'Model': ['MultinomialNB', 'LogisticRegression', 'SVC', 'RandomForestClassifier'],

    'Score': [acc_multi, acc_lr, acc_svc, acc_rfc]

})

Models = pd.DataFrame.from_dict(Models)

Models.sort_values(by = 'Score', ascending = False)