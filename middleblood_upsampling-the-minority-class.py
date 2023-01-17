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
data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

print(data.shape)

data.describe()

data.head()
print(data.isnull().sum())

print(data['Unnamed: 2'].unique())

print(data['Unnamed: 3'].unique())

print(data['Unnamed: 4'].unique())
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)

data.head()
data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

data.head()
print(data[data['target'] == 'ham'].count())

print(data[data['target'] == 'spam'].count())
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure()

data['target'].hist()

plt.title('Counts of spam and ham sms messages')

plt.xlabel('Message type')

plt.ylabel('Count')

plt.show()

from sklearn.model_selection import train_test_split

from sklearn.utils import resample



data['label'] = data['target'].map({ 'ham': 0, 'spam' : 1 })



# split to test and train sets

X = data['text']

y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#upsampling

data_minority = data[data.label == 1]

data_majority = data[data.label == 0]

print(data_minority.shape)

print(data_majority.shape)

majority_samples = data_majority.shape[0]



minority_upsampled = resample(data_minority, replace=True, n_samples=majority_samples, random_state=42)

data_upsampled = pd.concat([data_majority, minority_upsampled])

print(data_upsampled.shape)

print(data_upsampled['label'].value_counts())



X_upsampled = data_upsampled['text']

y_upsampled = data_upsampled['label']

X_train_upsampled, X_test_upsampled, y_train_upsampled, y_test_upsampled = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score



"""

Helper function for evaluating model performance.

Even though we compute multiple metrics, we will use f1_score as the metric to decide model performance as it incorporates both precision and recall.

"""

def evaluate(model, y_true, y_pred):

    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    p = precision_score(y_true, y_pred)

    r = recall_score(y_true, y_pred)

    roc = roc_auc_score(y_true, y_pred)

    print("Results for {0}:\n".format(model))

    print("precision={0},\nrecall={1},\nf1_score={2},\nroc_auc_score={3}".format(p, r, f1, roc))

    print("confusion matrix=\n{0}".format(cm))

    print("\n-----------------------------------------\n")
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

import time





def run_classification(X_train, X_test, y_train, y_test):

    classifiers = {

        'decision_tree' : DecisionTreeClassifier(),

        'random_forest' : RandomForestClassifier(),

        'svm' : SVC(),

        'multinomial_nb' : MultinomialNB(),

        'logistic_regression' : LogisticRegression()

    }



    for name, clf in classifiers.items():



        pipeline = Pipeline([

             ('count', CountVectorizer()),

             ('tfidf', TfidfTransformer()),

             (name, clf),

        ])



        parameters = {

            'count__ngram_range': [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],

            'count__stop_words': ['english', None],

            'tfidf__use_idf': (True, False),

        }



        grid = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)

        start = time.time()

        grid.fit(X_train, y_train)

        end = time.time()

        print("Time to train {0}: {1}".format(name, end - start))



        y_pred = grid.predict(X_test)

        evaluate(name, y_test, y_pred)

        #break

run_classification(X_train, X_test, y_train, y_test)
run_classification(X_train_upsampled, X_test_upsampled, y_train_upsampled, y_test_upsampled)