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
dftrain = pd.read_csv('../input/hselyc181018/train_lyc.csv', index_col='PassengerId')

dftest = pd.read_csv('../input/hselyc181018/test_lyc.csv', index_col='PassengerId')
dftrain
dftest
import tensorflow as tf

from tensorflow import feature_column, keras

tf.__version__
dftrain.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

dftest.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
age_mean = int(dftrain.Age.mean())

dftrain.Age.fillna(age_mean, inplace=True)

dftest.Age.fillna(age_mean, inplace=True)

dftrain.Embarked.fillna(dftrain.Embarked.mode().loc[0], inplace=True)
dfeval = dftrain.sample(frac=0.2, random_state=33)

dftrain.drop(dfeval.index, inplace=True)

y_train = dftrain.pop('Survived')

y_eval = dfeval.pop('Survived')
CATEGORICAL_COLUMNS = ['Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked']

NUMERIC_COLUMNS = ['Age', 'Fare']



feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:

  vocabulary = dftrain[feature_name].unique()

  feature_columns.append(feature_column.indicator_column(

      tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)))



for feature_name in NUMERIC_COLUMNS:

  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

feature_columns
def input_fn(features, labels, training=True, batch_size=64):

    """An input function for training or evaluating"""

    # Convert the inputs to a Dataset.

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))



    # Shuffle and repeat if you are in training mode.

    if training:

        dataset = dataset.shuffle(1000).repeat()

    

    return dataset.batch(batch_size)
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.

classifier = tf.estimator.DNNClassifier(

    feature_columns=feature_columns,

    # Two hidden layers of 30 and 10 nodes respectively.

    hidden_units=[32, 32, 16],

    # The model must choose between 3 classes.

    n_classes=2)
# Train the Model.

classifier.train(

    input_fn=lambda: input_fn(dftrain, y_train, training=True),

    steps=5000)
eval_result = classifier.evaluate(

    input_fn=lambda: input_fn(dfeval, y_eval, training=False))



print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
def input_fn(features, batch_size=10):

    """An input function for prediction."""

    # Convert the inputs to a Dataset without labels.

    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
dftest
dftest.isna().sum()
dftest.Embarked.fillna(dftrain.Embarked.mode().loc[0], inplace=True)
input_fn(dftest)
predictions = classifier.predict(

    input_fn=lambda: input_fn(dftest))

predictions
test_preds = []

for pred_dict in predictions:

    test_preds.append(pred_dict['class_ids'][0])
predictions = pd.Series(test_preds)

predictions
sub_df = pd.DataFrame(data={

    'PassengerId': dftest.index,

    'Survived': predictions

})

sub_df
sub_df.Survived.value_counts()
sub_df.to_csv('submission.csv', index=False)

print('done')