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
#import tensorflow

from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import datasets, layers, models



import matplotlib.pyplot as plt
# Load into pandas

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")





# Categorize string columns, drop unused columns

def categorize_objects(df):

    df['Pclass'] = pd.Categorical(df['Pclass'])

    df['Pclass'] = df.Pclass.cat.codes

    

    df['SibSp'] = pd.Categorical(df['SibSp'])

    df['SibSp'] = df.SibSp.cat.codes

    

    df['Parch'] = pd.Categorical(df['Parch'])

    df['Parch'] = df.Parch.cat.codes

    

    df['Sex'] = pd.Categorical(df['Sex'])

    df['Sex'] = df.Sex.cat.codes



    df['Ticket'] = pd.Categorical(df['Ticket'])

    df['Ticket'] = df.Ticket.cat.codes



    df['Cabin'] = pd.Categorical(df['Cabin'])

    df['Cabin'] = df.Cabin.cat.codes



    df['Embarked'] = pd.Categorical(df['Embarked'])

    df['Embarked'] = df.Embarked.cat.codes

    

    df['Age'] = pd.Categorical(df['Age'])

    df['Age'] = df.Age.cat.codes

    

    df.pop('Ticket')

    df.pop('Cabin')

    df.pop('Name')

    

    return df





# Apply categorization and split out ids/labels

test = categorize_objects(test)

train = categorize_objects(train)



test_ids = test.pop('PassengerId')

train_ids = train.pop('PassengerId')



labels = train.pop('Survived')



#normalize

#train = tf.keras.utils.normalize(train)

#test = tf.keras.utils.normalize(test)
print(train.dtypes)



print(train.shape)

print(labels.shape)



print (train["Pclass"].max())

print (train["Pclass"].min())



train
# create dataset and get batch

dataset = tf.data.Dataset.from_tensor_slices((train.values, labels.values))



train_dataset = dataset.shuffle(len(train)).batch(1)
# THE MODEL

model = tf.keras.Sequential([

    tf.keras.layers.Dense(10, input_dim=7, activation='relu'),

    #tf.keras.layers.Dense(10, activation='relu'),

    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam',

                loss=tf.keras.losses.BinaryCrossentropy(),

                metrics=['accuracy'])
# fit it

history = model.fit(train_dataset, epochs=15)
# Get predictions, format and output

predictions = model.predict(test)

test_labels = tf.math.round(predictions)



output_id = test_ids

label = test_labels.numpy().flatten()

      

submission = pd.DataFrame(np.array(list(zip(output_id,label))), columns=['PassengerId', 'Survived'])



submission[submission.isna().any(axis=1)]



print(gender_submission.shape)

print(submission.shape)



submission = submission.fillna(0)

submission = submission.astype(int)



submission.to_csv('submission.csv', index=False)