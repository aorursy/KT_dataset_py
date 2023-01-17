# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf #build mode
import matplotlib.pyplot as plt
from tensorflow.python.data import Dataset #organize input data
import functools
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#define data file path and columns
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'

CSV_COLUMNS_TRAIN = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
                'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
CSV_COLUMNS_TEST = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age',
                'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
DROP_COLUMNS = ['PassengerId', 'Cabin', 'Ticket']
TARGET_COLUMN = ['Survived']
def load_data():
    train_data = pd.read_csv(TRAIN_PATH, header=0,
                             names=CSV_COLUMNS_TRAIN)

    #shuffule the training data before split it to train and validation dataset
    train_data.reindex(np.random.permutation(train_data.index))

    test_data = pd.read_csv(TEST_PATH, header=0, names=CSV_COLUMNS_TEST)

    raw_data = [train_data, test_data]
    for data in raw_data:
        data.drop(columns=DROP_COLUMNS, inplace=True)
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        data.drop(columns=['Name'], inplace=True)
        if 'Survived' in data:
            data.dropna(subset=['Survived'], inplace=True)

    train_features, train_label = train_data, train_data.pop('Survived')
    return train_features, train_label, test_data
    train_features, train_label,  test_features = load_data()
    all_features = train_features.append(test_features)
    print(all_features.info())
    print(train_label.describe(include = 'all'))
    print(all_features.describe())
    print(all_features.head(10))
def input_fn(features, label, batch_size=10, epochs=None, shuffle=True):
    if label is None:
        inputs = dict(features)
    else:
        inputs = (dict(features), label)
    dataset = Dataset.from_tensor_slices(inputs)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
def construct_feature_columns(features):
    '''
    assemble feature columns
    '''
    return [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list('Pclass', [1,2,3])),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['male', 'female'])
        ),
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('Age'),
            boundaries= list(np.percentile(features['Age'], np.arange(10,100,20)))
        ),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity('SibSp', num_buckets=8, default_value=0)
        ),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity('Parch', num_buckets=5, default_value=0)
        ),
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('Fare'), boundaries=list(np.percentile(features['Fare'], np.arange(10,100,20)))
        ),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C', 'Q'])
        ),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list('Title', ['Mr', 'Mrs','Miss', 'Master'])
        )
    ]
def write_predict_csv(predicts):
    '''
    write predicts to gender_submission.csv
    '''
    test_data = pd.read_csv(TEST_PATH, header=0, names=CSV_COLUMNS_TEST)
    predict_df = pd.DataFrame()
    predict_df['PassengerId'] = test_data['PassengerId']
    predict_df['Survived'] = predicts
    predict_df.to_csv('gender_submission.csv', index=False)
    pass
    train_features, train_label, test_features = load_data()

    startIdx = int(len(train_features) * 0.75)

    train_input_fn = functools.partial(input_fn, train_features[:startIdx], train_label[:startIdx],
                                              batch_size=50)
    train_eval_input_fn = functools.partial(input_fn, train_features[:startIdx], train_label[:startIdx],
                                             epochs=1, shuffle=False)
    validation_input_fn = functools.partial(input_fn, train_features[startIdx:], train_label[startIdx:],
                                                            epochs=1, shuffle=False)
    test_input_fn = functools.partial(input_fn, test_features, None, epochs=1, shuffle=False)

    my_run_config = tf.estimator.RunConfig(save_checkpoints_secs=5)

    my_estimator = tf.estimator.DNNClassifier(
        hidden_units=[16, 16],
        feature_columns=construct_feature_columns(train_features),
        model_dir='modedir',
        config=my_run_config
    )

    steps_per_period = 5
    periods = 20
    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []

    for period in range(periods):

        my_estimator.train(train_input_fn, steps=steps_per_period)
        train_result = my_estimator.evaluate(train_eval_input_fn)
        validation_result = my_estimator.evaluate(validation_input_fn)
        train_losses.append(train_result['average_loss'])
        validation_losses.append(validation_result['average_loss'])
        train_accuracy.append(train_result['accuracy'])
        validation_accuracy.append(validation_result['accuracy'])
        print('train loss is {}, and val loss is {}'.format(train_result['average_loss'], validation_result['average_loss']))
        print('train acc is {}, and val acc is {}'.format(train_result['accuracy'], validation_result['accuracy']))

    predict_result = my_estimator.predict(test_input_fn)
    predicts = []
    for predict in predict_result:
        predicts.append(predict['probabilities'])
    predicts = np.argmax(predicts, axis=1)
    write_predict_csv(predicts)