# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy as sp 

import sklearn

import random 

import time 

from sklearn import preprocessing, model_selection

from keras.models import Sequential 

from keras.layers import Dense 

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

from sklearn.utils import shuffle

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn import preprocessing





from sklearn.linear_model import SGDClassifier

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')

test.head(5)
train.shape
train. describe()
train.info()
train.isnull().any()
test.isnull().any()
test_ids = test['Id']

test = test.drop(['Id'], 1)
le = preprocessing.LabelEncoder()

le.fit(train['class'])

print(list(le.classes_))



train['class'] = le.transform(train['class'])
train_ids = train['Id']

train = train.drop(['Id'], 1)
x_data = train.drop('class',axis=1)

y_labels = train['class']



X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=101)
buying = tf.feature_column.categorical_column_with_vocabulary_list("buying", ['high', 'low', 'med', 'vhigh'])

maintainence = tf.feature_column.categorical_column_with_vocabulary_list("maintainence", ['high', 'low', 'med', 'vhigh'])

doors = tf.feature_column.categorical_column_with_vocabulary_list("doors", ['3', '4', '5more', '2'])

persons = tf.feature_column.categorical_column_with_vocabulary_list("persons", ['4', 'more', '2'])

lug_boot = tf.feature_column.categorical_column_with_vocabulary_list("lug_boot", ['small', 'med', 'big'])

safety = tf.feature_column.categorical_column_with_vocabulary_list("safety", ['low', 'med', 'high'])
feat_cols = [buying, maintainence, doors, persons, lug_boot, safety]
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,

                                               y=y_train,

                                               batch_size=500,

                                               num_epochs=10000,

                                               shuffle=True)
input_func_test = tf.estimator.inputs.pandas_input_fn(x=test,

                                               num_epochs=500,

                                               shuffle=False)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=4)
model.train(input_fn=input_func,steps=5000)
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
predictions = list(model.predict(input_fn=pred_fn))

probs = pd.Series([pred['class_ids'][0] for pred in predictions])
from sklearn.metrics import classification_report



final_preds = []

for pred in predictions:

    final_preds.append(pred['class_ids'][0])



print(classification_report(y_test,final_preds))
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=len(X_test),shuffle=False)
results = model.evaluate(eval_input_func)

results
pred_fn_test = tf.estimator.inputs.pandas_input_fn(x=test, batch_size=len(test), shuffle=False)
predictions_test = list(model.predict(input_fn=pred_fn_test))

probs_test = pd.Series([pred['class_ids'][0] for pred in predictions_test])
print(type(probs_test))

print(len(probs_test))
preds_test_1 = []

for pred in predictions_test:

    preds_test_1.append(pred['class_ids'][0])



print(len(preds_test_1))
preds_test = le.inverse_transform(preds_test_1)

print(type(preds_test))
df = pd.DataFrame(columns=['Id', 'Class_vgood', 'Class_good', 'Class_acc', 'Class_unacc'])



for i, ids, preds in zip(range(len(test_ids)), test_ids, preds_test):

    

    if(preds == 'vgood'):

        submission = pd.DataFrame({

            "Id": ids,

            "Class_vgood": 1,

            "Class_good": 0,

            "Class_acc": 0,

            "Class_unacc": 0,

        }, index=[i])

        df = df.append(submission)

        

    if(preds == 'good'):

        submission = pd.DataFrame({

            "Id": ids,

            "Class_vgood": 0,

            "Class_good": 1,

            "Class_acc": 0,

            "Class_unacc": 0,

        }, index=[i])

        df = df.append(submission)

        

    if(preds == 'acc'):

        submission = pd.DataFrame({

            "Id": ids,

            "Class_vgood": 0,

            "Class_good": 0,

            "Class_acc": 1,

            "Class_unacc": 0,

        }, index=[i])

        df = df.append(submission)

        

    if(preds == 'unacc'):

        submission = pd.DataFrame({

            "Id": ids,

            "Class_vgood": 0,

            "Class_good": 0,

            "Class_acc": 0,

            "Class_unacc": 1,

        }, index=[i])

        df = df.append(submission)
df.to_csv('sampleSubmission.csv', index=False)