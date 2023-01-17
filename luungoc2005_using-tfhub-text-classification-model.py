# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load data
seed = 197

import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
print(tf.__version__)
import pandas as pd

train_df = pd.read_csv('../input/train.tsv',  sep="\t")
test_df = pd.read_csv('../input/test.tsv',  sep="\t")
train_df.head()
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], shuffle=False)
embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1")
estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))
steps = 10000
estimator.train(input_fn=train_input_fn, steps=steps)
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
sub = pd.read_csv('../input/sampleSubmission.csv')
sub.head()
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["Phrase"], shuffle=False)

def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

sub['Sentiment'] = get_predictions(estimator, predict_test_input_fn)

sub.to_csv('sub_tfhub.csv', index=False)
sub.head()