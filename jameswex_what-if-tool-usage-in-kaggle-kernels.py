!pip install witwidget

!jupyter nbextension install --py --symlink --sys-prefix witwidget

!jupyter nbextension enable --py --sys-prefix witwidget
import pandas as pd

csv_columns = [

  "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital-Status",

  "Occupation", "Relationship", "Race", "Sex", "Capital-Gain", "Capital-Loss",

  "Hours-per-week", "Country", "Over-50K"]

df = pd.read_csv("../input/adult-training.csv", names=csv_columns, skipinitialspace=True)

df
import pandas as pd

import numpy as np

import tensorflow as tf

import functools



# Creates a tf feature spec from the dataframe and columns specified.

def create_feature_spec(df, columns=None):

    feature_spec = {}

    if columns == None:

        columns = df.columns.values.tolist()

    for f in columns:

        if df[f].dtype is np.dtype(np.int64):

            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.int64)

        elif df[f].dtype is np.dtype(np.float64):

            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.float32)

        else:

            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.string)

    return feature_spec



# Creates simple numeric and categorical feature columns from a feature spec and a

# list of columns from that spec to use.

#

# NOTE: Models might perform better with some feature engineering such as bucketed

# numeric columns and hash-bucket/embedding columns for categorical features.

def create_feature_columns(columns, feature_spec):

    ret = []

    for col in columns:

        if feature_spec[col].dtype is tf.int64 or feature_spec[col].dtype is tf.float32:

            ret.append(tf.feature_column.numeric_column(col))

        else:

            ret.append(tf.feature_column.indicator_column(

                tf.feature_column.categorical_column_with_vocabulary_list(col, list(df[col].unique()))))

    return ret



# An input function for providing input to a model from tf.Examples

def tfexamples_input_fn(examples, feature_spec, label, mode=tf.estimator.ModeKeys.EVAL,

                       num_epochs=None, 

                       batch_size=64):

    def ex_generator():

        for i in range(len(examples)):

            yield examples[i].SerializeToString()

    dataset = tf.data.Dataset.from_generator(

      ex_generator, tf.dtypes.string, tf.TensorShape([]))

    if mode == tf.estimator.ModeKeys.TRAIN:

        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

    dataset = dataset.batch(batch_size)

    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, label, feature_spec))

    dataset = dataset.repeat(num_epochs)

    return dataset



# Parses Tf.Example protos into features for the input function.

def parse_tf_example(example_proto, label, feature_spec):

    parsed_features = tf.parse_example(serialized=example_proto, features=feature_spec)

    target = parsed_features.pop(label)

    return parsed_features, target



# Converts a dataframe into a list of tf.Example protos.

def df_to_examples(df, columns=None):

    examples = []

    if columns == None:

        columns = df.columns.values.tolist()

    for index, row in df.iterrows():

        example = tf.train.Example()

        for col in columns:

            if df[col].dtype is np.dtype(np.int64):

                example.features.feature[col].int64_list.value.append(int(row[col]))

            elif df[col].dtype is np.dtype(np.float64):

                example.features.feature[col].float_list.value.append(row[col])

            elif row[col] == row[col]:

                example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))

        examples.append(example)

    return examples



# Converts a dataframe column into a column of 0's and 1's based on the provided test.

# Used to force label columns to be numeric for binary classification using a TF estimator.

def make_label_column_numeric(df, label_column, test):

  df[label_column] = np.where(test(df[label_column]), 1, 0)
import numpy as np



# Set the column in the dataset you wish for the model to predict

label_column = 'Over-50K'



# Make the label column numeric (0 and 1), for use in our model.

# In this case, examples with a target value of '>50K' are considered to be in

# the '1' (positive) class and all other examples are considered to be in the

# '0' (negative) class.

make_label_column_numeric(df, label_column, lambda val: val == '>50K')



# Set list of all columns from the dataset we will use for model input.

input_features = [

  'Age', 'Workclass', 'Education', 'Marital-Status', 'Occupation',

  'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss',

  'Hours-per-week', 'Country']



# Create a list containing all input features and the label column

features_and_labels = input_features + [label_column]



# Convert data to example format

examples = df_to_examples(df)



# Create a feature spec for the classifier

feature_spec = create_feature_spec(df, features_and_labels)



# Define and train the classifier

num_steps = 5000

train_inpf = functools.partial(tfexamples_input_fn, examples, feature_spec, label_column)

classifier = tf.estimator.LinearClassifier(

    feature_columns=create_feature_columns(input_features, feature_spec))

classifier.train(train_inpf, steps=num_steps)
from witwidget.notebook.visualization import WitConfigBuilder

from witwidget.notebook.visualization import WitWidget



# Setup the tool with some examples and the trained classifier

config_builder = WitConfigBuilder(examples[0:2000]).set_estimator_and_feature_spec(

    classifier, feature_spec)

WitWidget(config_builder, height=800)