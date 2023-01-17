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
input_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

x_data = input_df.text.values

y_data = input_df.target.values
from sklearn.model_selection import train_test_split

train_data,test_data, train_true,test_true = train_test_split(x_data, y_data, test_size = 0.20, random_state = 0)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english',max_features=10000)

train_vector = vectorizer.fit_transform(train_data)

test_vector = vectorizer.transform(test_data)
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn import metrics
# Logistic Regression



logreg = linear_model.LogisticRegression()

logreg.fit(train_vector, train_true)

test_pred = logreg.predict(test_vector)

print(metrics.classification_report(test_true, test_pred))
# Support Vector Machines



svc = svm.SVC(kernel = 'rbf', gamma=0.085 ,random_state = 0)

svc.fit(train_vector, train_true)

test_pred = svc.predict(test_vector)

print(metrics.classification_report(test_true, test_pred))
knn = neighbors.KNeighborsClassifier(n_neighbors = 3)

knn.fit(train_vector, train_true)

test_pred = knn.predict(test_vector)

print(metrics.classification_report(test_true, test_pred))
# Decision Tree



decision_tree = tree.DecisionTreeClassifier()

decision_tree.fit(train_vector, train_true)

test_pred = decision_tree.predict(test_vector)

print(metrics.classification_report(test_true, test_pred))
# Random Forest



random_forest = ensemble.RandomForestClassifier(n_estimators=200)

random_forest.fit(train_vector, train_true)

test_pred = random_forest.predict(test_vector)

print(metrics.classification_report(test_true, test_pred))
input_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english',max_features=10000)

train_vector = vectorizer.fit_transform(input_df.text.values)

train_true = input_df.target.values

test_vector = vectorizer.transform(test_df.text.values)



# Logistic Regression



logreg = linear_model.LogisticRegression()

logreg.fit(train_vector, train_true)

test_pred = logreg.predict(test_vector)

test_df = test_df.drop(['keyword','location','text'], axis=1)

test_df['target'] = test_pred
test_df.to_csv("nlp_crash_course_submission.csv", index = False)