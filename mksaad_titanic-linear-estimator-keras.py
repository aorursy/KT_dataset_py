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
import tensorflow.compat.v2.feature_column as fc



import tensorflow as tf

tf.__version__
# Load dataset.

dftrain = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

dftest = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
dftrain
dftest
dftrain.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

dftest.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
age_mean = int(dftrain.Age.mean())

age_mean
dftrain.Age.fillna(age_mean, inplace=True)

dftest.Age.fillna(age_mean, inplace=True)
print(dftrain.isna().sum())
dftrain.Embarked.mode().loc[0]
dftrain.Embarked.fillna(dftrain.Embarked.mode().loc[0], inplace=True)
print(dftest.isna().sum())
dftest.Fare.fillna(int(dftrain.Fare.mean()), inplace=True)
dfeval = dftrain.sample(frac=0.2, random_state=33)

dftrain.drop(dfeval.index, inplace=True)

y_train = dftrain.pop('Survived')

y_eval = dfeval.pop('Survived')
dftrain.keys()


CATEGORICAL_COLUMNS = ['Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked']

NUMERIC_COLUMNS = ['Age', 'Fare']



feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:

  vocabulary = dftrain[feature_name].unique()

  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))



for feature_name in NUMERIC_COLUMNS:

  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

feature_columns
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

  print('A batch of class:', feature_batch['Pclass'].numpy())

  print()

  print('A batch of Labels:', label_batch.numpy())
age_x_gender = tf.feature_column.crossed_column(['Age', 'Sex'], hash_bucket_size=100)

derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)

linear_est.train(train_input_fn)

result = linear_est.evaluate(eval_input_fn)

result
dftest.shape
y_test = pd.Series( [-1] * dftest.shape[0] )

y_test
test_input_fn = make_input_fn(dftest, y_test, num_epochs=1, shuffle=False)
pred_dicts = list(linear_est.predict(test_input_fn))

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs
predictions = []

for p in probs:

    if p < 0.5:

        predictions.append(0)

    else:

        predictions.append(1)

predictions = pd.Series(predictions)

predictions
sub_df = pd.DataFrame(data={

    'PassengerId': dftest.index,

    'Survived': predictions

})

sub_df
sub_df.Survived.value_counts()
sub_df.to_csv('submission.csv', index=False)