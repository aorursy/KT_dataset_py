from __future__ import unicode_literals, print_function

from itertools import chain

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import numpy as np

from numpy import random

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from gensim import parsing

from sklearn.metrics import classification_report

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from gensim.models import Doc2Vec

from sklearn import utils

import gensim

from gensim.models.doc2vec import TaggedDocument

from sklearn.svm import LinearSVC

from sklearn.datasets import make_classification

from sklearn.metrics import confusion_matrix

import itertools
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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.shape
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test['Sex'].replace(['female','male'],[0,1],inplace=True)

df_tests = df_test[["Pclass", "Sex", "SibSp", "Parch"]]

df_tests.head()
df_train.head()
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
df_trains = df_train[["Pclass", "Sex", "SibSp", "Parch","Survived"]]
df_trains
df_trains = pd.get_dummies(df_trains)
missing_val_count_by_column = (df_trains.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# from sklearn.impute import SimpleImputer

# my_imputer = SimpleImputer()

# df_trains = my_imputer.fit_transform(df_trains)
df_trains.head()
print('Proportion of the classes in the data:')

print(df_trains['Survived'].value_counts() / len(df_trains))
plt.figure(figsize=(10,4))

df_trains['Survived'].value_counts().sort_index().plot.bar(color=['blue', 'red'])
def plot_confusion_matrix(cm, classes,

                          normalize=True,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
my_categories=['0','1']



# #Seperate data into feature and results

X, y = df_trains.loc[:, df_trains.columns != 'Survived'], df_trains['Survived']



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# instantiate the model (using the default parameters)

logreg = LogisticRegression()



logreg.fit(X,y)



#

y_pred=logreg.predict(df_tests)



output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

# lrm1 = accuracy_score(y_pred, y_test)



# print('accuracy %s' % lrm1)

# print(classification_report(y_test, y_pred,target_names=my_categories))



# # Compute confusion matrix

# cnf_matrix = confusion_matrix(y_test, y_pred,labels=my_categories)

# np.set_printoptions(precision=2)



# # Plot non-normalized confusion matrix

# plt.figure()

# plot_confusion_matrix(cnf_matrix, classes=['Death(0)','Alive(1)']

#                       ,normalize= True,  title='Confusion matrix')
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

model.fit(X_train, y_train)

predictions = model.predict(df_tests)



output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

# rfc = accuracy_score(predictions, y_test)

# print('accuracy %s' % rfc)

# print(classification_report(y_test, predictions,target_names=my_categories))