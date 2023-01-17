# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import math

import random

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



import matplotlib.pyplot as plt

plt.close('all')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_test = pd.read_csv("../input/train.csv")

msk = np.random.rand(len(train_test)) < 0.8

train = train_test[msk].copy()

test = train_test[~msk].copy()

train_y = train.pop('Survived')

test_y = test.pop('Survived')



submit = pd.read_csv("../input/test.csv")



for df in [train, test, submit]:

    # normalize Pclass to [0, 1, 2]

    df['Pclass'] = df['Pclass'] - 1



    # If no age, pick randomly between 20-40

    df['Age'] = df['Age'].map(lambda x: round(random.random() * 20 + 20) if math.isnan(x) else x)

    

    # If no fare, pick randomly between 7.25-11.25

    df['Fare'] = df['Fare'].map(lambda x: round(random.random() * 4 + 7) if math.isnan(x) else x)



    # If no embarked, choose 'C'

    df['Embarked'] = df['Embarked'].map(lambda x: 'C' if not isinstance(x, str) else x)



    # If no cabin, set ''

    df['Cabin'] = df['Cabin'].map(lambda x: 'XX' if not isinstance(x, str) else x[0])

    

    # ignore ticket # and cabin

    df.pop('Ticket')



# check for NaN

for df in [train, test, submit]:

    assert len(df[df.isnull().any(axis=1)]) == 0

    

train.head(30)
train['Cabin'].unique()
# Define features

feature_columns = [

    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(

        key='Pclass',

        num_buckets=3)),

    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(

        key='Sex',

        vocabulary_list=['male', 'female'])),

    tf.feature_column.bucketized_column(

        source_column=tf.feature_column.numeric_column('Age'),

        boundaries=[7, 20, 40]),

    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(

        key='Cabin',

        vocabulary_list=['XX', 'C', 'E', 'G', 'D', 'A', 'B', 'F'])),

    tf.feature_column.numeric_column(key='SibSp'),

    tf.feature_column.numeric_column(key='Parch'),

]



dnn = tf.estimator.DNNClassifier(

    feature_columns=feature_columns,

    hidden_units=[4, 4],

    dropout=0.2,

    n_classes=2)



dnn_larger = tf.estimator.DNNClassifier(

    feature_columns=feature_columns,

    hidden_units=[8, 4],

    dropout=0.2,

    n_classes=2)



# LinearClassifer with default optimizer

# linear_classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns_fare_cabin)



estimators = [

    (dnn, 'dnn'),

    (dnn_larger, 'dnn_larger'),  

]
# Train and evaluate function

def input_fn(features, labels, training=True, batch_size=256):

    """An input function for training or evaluating"""

    # Convert the inputs to a Dataset.

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))



    # Shuffle and repeat if you are in training mode.

    if training:

        dataset = dataset.shuffle(1000).repeat()

    

    return dataset.batch(batch_size)



# Do a train and evaluate loop for each estimator

epochs = 40

step_incr = 25

steps = [step_incr * i for i in range(1, epochs+1)]

results = []

for classifier, classifier_name in estimators:

    train_results = []

    test_results = []

    for _ in range(epochs):

        classifier.train(

            input_fn=lambda: input_fn(train, train_y, training=True),

            steps=step_incr)

        train_results.append(classifier.evaluate(

            input_fn=lambda: input_fn(train, train_y, training=False))['accuracy'])

        test_results.append(classifier.evaluate(

            input_fn=lambda: input_fn(test, test_y, training=False))['accuracy'])

    results.append((train_results, classifier_name + '-train'))

    results.append((test_results, classifier_name + '-test'))



column_names = [r[1] for r in results]

results = [r[0] for r in results]

data = np.swapaxes(np.array(results), 0, 1)

df = pd.DataFrame(data, index=steps, columns=column_names)

plt.figure()

df.plot(figsize=(10, 10))
df
# Generate predictions from the model



def predict_input_fn(features, batch_size=256):

    """An input function for prediction."""

    # Convert the inputs to a Dataset without labels.

    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)



for classifier, classifier_name in estimators:

    predictions = classifier.predict(

        input_fn=lambda: predict_input_fn(submit))



    submission = pd.DataFrame(data={

        'PassengerId': [x for x in submit.iloc[:,0]],

        'Survived': [int(p['classes'][0]) for p in predictions]

    })



    submission.to_csv(classifier_name + '-submission.csv', index=False)



submission.head(20)