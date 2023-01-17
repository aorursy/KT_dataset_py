import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submit = test.copy()

submit = submit.drop(['Name','Fare','Ticket','Cabin','Embarked','Pclass','Sex','Age','SibSp','Parch'],axis=1)
print(train.shape)

print(test.shape)

train = train.fillna(0)

test = test.fillna(0)
target = train['Survived']

train = train.drop(['Survived','Name','PassengerId','Ticket','Cabin','Embarked'],axis=1)

train['Sex'] = train['Sex'].replace(['male','female'],['0','1']).to_numpy()

test = test.drop(['Name','PassengerId','Ticket','Cabin','Embarked'],axis=1)

test['Sex'] = test['Sex'].replace(['male','female'],['0','1']).to_numpy()
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, make_scorer

from statistics import mean
tree = DecisionTreeClassifier(random_state=0)

gnb = GaussianNB()

classifier = [tree,gnb]
def testResultClassifier(classifier, X, Y):

  cv = KFold(n_splits=5, shuffle=False)

  index=['Fold_1','Fold_2','Fold_3','Fold_4','Fold_5']

  df_scores = []

  f1_average = []

  for model in classifier:

    recalls,precisions,f1 = [],[],[]

    for train_index, test_index in cv.split(X):

      X_train, X_test = X.iloc[train_index], X.iloc[test_index]

      Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

      model.fit(X_train,Y_train)

      predictions = model.predict(X_test)

      recalls.append(recall_score(Y_test, predictions))

      precisions.append(precision_score(Y_test, predictions))

      f1.append(f1_score(Y_test, predictions))

    score = {'recall_score':recalls,'precision_score':precisions,'f1_score':f1}

    df_scores.append(pd.DataFrame(score, index))

    f1_average.append(mean(f1))

  return df_scores, f1_average
score,f1_avg = testResultClassifier(classifier,train,target)
import tensorflow as tf
def testResultNN(X, Y):

  cv = KFold(n_splits=5, shuffle=False)

  index=['Fold_1','Fold_2','Fold_3','Fold_4','Fold_5']

  recalls,precisions,f1 = [],[],[]

  df_scores = []

  f1_average = []

  for train_index, test_index in cv.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    X_train = np.asarray(X_train).astype(np.float32)

    Y_train = np.asarray(Y_train).astype(np.float32)

    X_test = np.asarray(X_test).astype(np.float32)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(512, input_shape=(6,)))

    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(256))

    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(2))

    model.add(tf.keras.layers.Activation('sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 

              loss=tf.keras.losses.sparse_categorical_crossentropy)

    model.fit(X_train, Y_train, epochs=100)

    predictions = model.predict(X_test)

    predictions = predictions.argmax(axis=1)

    recalls.append(recall_score(Y_test, predictions))

    precisions.append(precision_score(Y_test, predictions))

    f1.append(f1_score(Y_test, predictions))

  score = {'recall_score':recalls,'precision_score':precisions,'f1_score':f1}

  df_scores.append(pd.DataFrame(score, index))

  f1_average.append(mean(f1))

  return df_scores, f1_average
nn_score, nn_f1 = testResultNN(train,target)
score[0]
print('Average F-Measure:',f1_avg[0])
score[1]
print('Average F-Measure:',f1_avg[1])
nn_score[0]
print('Average F-Measure:',nn_f1[0])