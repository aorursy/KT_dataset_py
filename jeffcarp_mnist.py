import csv



import numpy as np

import pandas as pd

import tensorflow as tf



from subprocess import check_output



from tensorflow.contrib.learn.python.learn.estimators import run_config

from tensorflow.contrib.training.python.training import hparam

from sklearn.model_selection import train_test_split



PIXELS = 784

NUM_EPOCHS = 50

TRAIN_BATCH_SIZE = 40

EVAL_BATCH_SIZE = 40

TRAIN_TEST_SPLIT_RANDOM_STATE = 42



INPUT_COLUMNS = [

  tf.feature_column.numeric_column(key='pixels', shape=(PIXELS,)),

]
config = run_config.RunConfig()

hparams = hparam.HParams(

    num_epochs=NUM_EPOCHS, 

    train_batch_size=TRAIN_BATCH_SIZE, 

    eval_batch_size=EVAL_BATCH_SIZE,

    eval_steps=100

)
# Load training data

df_train = pd.read_csv('../input/train.csv')

df_train.describe()
df_test = pd.read_csv('../input/test.csv')

df_test.describe()
X = df_train.loc[:, 'pixel0':'pixel783']

y = df_train['label']



X_train, X_eval, y_train, y_eval = train_test_split(X, y, 

    test_size=0.25, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)
experiment = tf.contrib.learn.Experiment(

    tf.contrib.learn.DNNClassifier(

        config=config,

        n_classes=10,

        feature_columns=INPUT_COLUMNS,

        hidden_units=[1024, 512, 256],

    ),

    train_input_fn=tf.estimator.inputs.numpy_input_fn(

        x={'pixels': X_train.as_matrix()},

        y=y_train.as_matrix(),

        num_epochs=hparams.num_epochs,

        batch_size=hparams.train_batch_size,

        shuffle=True

    ),

    eval_delay_secs=1,

    eval_input_fn=tf.estimator.inputs.numpy_input_fn(

        x={'pixels': X_eval.as_matrix()},

        y=y_eval.as_matrix(),

        num_epochs=None,

        batch_size=hparams.eval_batch_size,

        shuffle=False # Don't shuffle evaluation data

    )

)
experiment.train()
experiment.evaluate()
X_test = df_test.loc[:, 'pixel0':'pixel783']



predict_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={'pixels': X_test.as_matrix()},

    num_epochs=1,

    batch_size=hparams.eval_batch_size,

    shuffle=False # Don't shuffle evaluation data

)

predictions = experiment.estimator.predict_classes(input_fn=predict_input_fn)



with open('submission.csv', 'w') as csvfile:

    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(['ImageId', 'Label'])

    

    for index, label in enumerate(predictions):

        csvwriter.writerow([index+1, label])