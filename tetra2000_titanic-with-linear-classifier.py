# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dftrain = pd.read_csv("/kaggle/input/titanic/train.csv", index_col=0, dtype={"Cabin":"string", "Embarked": "string"})

# FIXME: tf.data.Dataset.from_tensor_slices raises error with some columns

dftrain = dftrain.drop(columns=["Cabin", "Embarked"]).dropna()

y_train = dftrain.pop("Survived")



dftest = pd.read_csv("/kaggle/input/titanic/test.csv", index_col=0)
# DEBUG ONLY

# Use gender_submission as testing value

dfsubm = pd.read_csv("/kaggle/input/titanic/gender_submission.csv", index_col=0)

dfeval = dftest.join(dfsubm)

# FIXME: tf.data.Dataset.from_tensor_slices raises error with some columns

dfeval = dfeval.drop(columns=["Cabin", "Embarked"]).dropna()

y_eval = dfeval.pop("Survived")
dftrain.head()
y_train.head()
dftrain.Age.hist(bins=20)
dftrain.Sex.value_counts().plot(kind='barh')
# dftrain["Embarked"].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('Sex').Survived.mean().plot(kind='barh').set_xlabel('% survive')
# CATEGORY_COLUMNS = ["Sex", "Cabin", "Embarked"]

CATEGORY_COLUMNS = ["Sex"]

NUMERIC_CULUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]



feature_columns = []



for feature_name in CATEGORY_COLUMNS:

    vocabulary = dftrain[column].unique()

    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))



for feature_name in NUMERIC_CULUMNS:

    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):

    def input_function():

        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:

            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds

    return input_function



train_input_fn = make_input_fn(dftrain, y_train)

eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
ds = make_input_fn(dftrain, y_train, batch_size=10)()

for feature_batch, label_batch in ds.take(1):

  print('Some feature keys:', list(feature_batch.keys()))

  print()

  print('A batch of class:', feature_batch['Fare'])

  print()

  print('A batch of Labels:', label_batch)
feature_columns

age_column = feature_columns[2]

tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)

result = linear_est.evaluate(eval_input_fn)



print(result)
test_input_fn = lambda: tf.data.Dataset.from_tensor_slices(dict(dftest.drop(columns=["Cabin", "Embarked"]))).batch(32)

result = linear_est.predict(test_input_fn)



predictions = []

for i in result:

    predictions.append(i)





    

i = 0

passenger_id_list = []

survived_list = []

for df_index, row in dftest.iterrows():

    passenger_id_list.append(df_index)

    survived_list.append(int(predictions[i]["classes"][0]))

    

    i += 1



d = {'PassengerId': passenger_id_list, 'Survived': survived_list}

df = pd.DataFrame(data=d)

df.to_csv("out.csv", index=False)