# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from IPython.display import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn



# Any results you write to the current directory are saved as output.
train_df = pd.read_json('../input/whats-cooking-kernels-only/train.json')

test_df = pd.read_json('../input/whats-cooking-kernels-only/test.json')
train_df['cuisine'].value_counts()
plt.figure(figsize=(20,10))

sns.countplot(train_df['cuisine'])
all_data = pd.concat([train_df.drop(['cuisine'], axis=1), test_df], axis=0)
all_data.columns, all_data.shape
#Just get all ingredients

all_ingredients = np.concatenate([data for data in all_data['ingredients']])

all_ingredients = pd.Series(all_ingredients)
for data in all_ingredients.unique():

    all_data[data] = all_data['ingredients'].apply(lambda x: 1 if data in x else 0)
all_data.shape
X = all_data[:39774]

X_test = all_data[39774:]
X.drop(['id', 'ingredients'], axis=1, inplace=True)

X_test.drop(['id', 'ingredients'], axis=1, inplace=True)
y = train_df['cuisine']
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

scores = cross_val_score(clf, X, y, cv=5)

print('Cross validation mean scores %.3f' %scores.mean())
tf = TfidfVectorizer(binary=True)

X_train_text = list(train_df['ingredients'].apply(lambda x: ' '.join(x)))

X = tf.fit_transform(X_train_text)
X.shape
scores = cross_val_score(clf, X, y, cv=5)

print('Cross validation mean scores %.3f' %scores.mean())
from IPython.display import Image

Image("../input/sklearn-algorithm-cheat-sheet/ml_map.png")
model = SVC(C=100, kernel='rbf', 

            gamma=1, 

            coef0=1, 

            shrinking=True, 

            tol=0.001, 

            probability=False,

            cache_size=200,

            class_weight=None,

            verbose=False,

            max_iter=-1,

            decision_function_shape=None,

            random_state=None)
scores = cross_val_score(model, X, y, cv=5)

print('Cross validation mean scores %.3f' %scores.mean())
# predict_result = clf.predict(X_test)

# submit_result = pd.concat([test_df['id'], pd.Series(predict_result)], axis=1)

# submit_result.columns=['id', 'cuisine']

# submit_result.to_csv('sample_submission.csv', index=False)