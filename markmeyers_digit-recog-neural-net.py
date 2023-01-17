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



# Any results you write to the current directory are saved as output.

#Validate working directory

os.getcwd()

print(os.getcwd())

#Validate Current Path and create Path to data

from pathlib import Path

INPUT = Path("../input/digit-recognizer")

os.listdir(INPUT)



#Import CSV into Pandas dataframe and test shape of file

train_df = pd.read_csv(INPUT/"train.csv")

print(train_df.shape)
import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import os



#suppress tf.logging

import logging

logging.getLogger('tensorflow').disabled = True

   

import tensorflow as tf

from datetime import datetime



from tensorflow.python.framework.ops import disable_eager_execution



disable_eager_execution()
#Split training data

from sklearn.model_selection import train_test_split



#Create new test split / rerun with new seed

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['label'], axis=1), train_df["label"], shuffle=True,

train_size=.85, random_state=1)
train_df.label.value_counts().sort_index().plot(kind="bar")

plt.title("Distribution of Digits in dataset")

plt.show()
Reduction = y_train.value_counts().sort_index() / train_df.label.value_counts().sort_index()



#visualize represented label quantity to ensure adequate representation of ALL labels

Reduction.plot(kind="bar")

axes = plt.gca()

axes.set_ylim([0.83,0.87])

plt.title("Reduction via train/test split of Digits in original vs training")

plt.show()
X_train = X_train.astype(np.float32)

X_test = X_test.astype(np.float32)

y_train = y_train.astype(np.int32)

y_test = y_test.astype(np.int32)
# Check the shape of the trainig data set array

print('Shape of X_train_data:', X_train.shape)

print('Shape of y_train_data:', y_train.shape)

print('Shape of X_test_data:', X_test.shape)

print('Shape of y_test_data:', y_test.shape)
X_train = X_train.to_numpy() 

X_test = X_test.to_numpy()

y_train = y_train.to_numpy()

y_test = y_test.to_numpy()
X_train = X_train.reshape((35700, 28, 28, 1))

X_test = X_test.reshape((6300, 28, 28, 1))

#y_train, y_test = y_train / 255.0, y_test / 255.0
X_train = X_train.astype(np.float32)/ 255.0

X_test = X_test.astype(np.float32)/ 255.0

y_train = y_train.astype(np.int32)

y_test = y_test.astype(np.int32)
# Check the shape of the trainig data set array

print('Shape of X_train_data:', X_train.shape)

print('Shape of y_train_data:', y_train.shape)

print('Shape of X_test_data:', X_test.shape)

print('Shape of y_test_data:', y_test.shape)
#Model 2a - 2D with 10 Neurons each layer

start=datetime.now()

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]

dnn_clf_2a = tf.estimator.DNNClassifier(hidden_units=[10,10], n_classes=10,

                                     feature_columns=feature_cols)



input_fn_2a = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_train}, y=y_train, num_epochs=50, batch_size=50, shuffle=True)

dnn_clf_2a.train(input_fn=input_fn_2a)

time1 = print("Time to calc:", datetime.now()-start)

time1
#evaluate training accuracy (2a)

train_2a_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_train}, y=y_train, shuffle=False)

eval_results_train_2a = dnn_clf_2a.evaluate(input_fn=train_2a_input_fn)

eval_results_train_2a
#evaluate test accuracy (2a)

test_2a_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_test}, y=y_test, shuffle=False)

eval_results_test_2a = dnn_clf_2a.evaluate(input_fn=test_2a_input_fn)

eval_results_test_2a
#confusion matrix for 2a predictions

raw_predictions_2a = dnn_clf_2a.predict(input_fn=test_2a_input_fn)

predictions_2a = [p['class_ids'][0] for p in raw_predictions_2a]

confusion_matrix_2a = tf.math.confusion_matrix(y_test, predictions_2a)



with tf.compat.v1.Session():

    con_mat = tf.Tensor.eval(confusion_matrix_2a,feed_dict=None, session=None)
classes=[0,1,2,3,4,5,6,7,8,9]

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

 

con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)
figure = plt.figure(figsize=(6, 5))

sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.coolwarm)

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title('(1) 2 layer model with 10 Neurons')

plt.show()
#Model 2b - 2D with 50 Neurons each layer

start=datetime.now()

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]

dnn_clf_2b = tf.estimator.DNNClassifier(hidden_units=[150,50], n_classes=10,

                                     feature_columns=feature_cols)



input_fn_2b = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_train}, y=y_train, num_epochs=50, batch_size=50, shuffle=True)

dnn_clf_2b.train(input_fn=input_fn_2b)

time2 = print("Time to calc:", datetime.now()-start)

time2
#evaluate training accuracy (2b)

train_2b_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_train}, y=y_train, shuffle=False)

eval_results_train_2b = dnn_clf_2a.evaluate(input_fn=train_2b_input_fn)

eval_results_train_2b
#evaluate test accuracy (2b)

test_2b_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_test}, y=y_test, shuffle=False)

eval_results_test_2b = dnn_clf_2b.evaluate(input_fn=test_2a_input_fn)

eval_results_test_2b
#confusion matrix for 2a predictions

raw_predictions_2b = dnn_clf_2b.predict(input_fn=test_2b_input_fn)

predictions_2b = [p['class_ids'][0] for p in raw_predictions_2b]

confusion_matrix_2b = tf.math.confusion_matrix(y_test, predictions_2b)



with tf.compat.v1.Session():

    con_mat = tf.Tensor.eval(confusion_matrix_2b,feed_dict=None, session=None)
classes=[0,1,2,3,4,5,6,7,8,9]

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

 

con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)
figure = plt.figure(figsize=(6, 5))

sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.coolwarm)

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title('(2) 2 layer model with mixed neurons')

plt.show()
#Model 5a - 5D with 20 Neurons each layer

start=datetime.now()

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]

dnn_clf_5a = tf.estimator.DNNClassifier(hidden_units=[20,20,20,20,20], n_classes=10,

                                     feature_columns=feature_cols)



input_fn_5a = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_train}, y=y_train, num_epochs=50, batch_size=50, shuffle=True)

dnn_clf_5a.train(input_fn=input_fn_5a)

time3 = print("Time to calc:", datetime.now()-start)

time3
#evaluate training accuracy (5a)

train_5a_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_train}, y=y_train, shuffle=False)

eval_results_train_5a = dnn_clf_5a.evaluate(input_fn=train_5a_input_fn)

eval_results_train_5a
#evaluate test accuracy (5a)

test_5a_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_test}, y=y_test, shuffle=False)

eval_results_test_5a = dnn_clf_5a.evaluate(input_fn=test_5a_input_fn)

eval_results_test_5a
#confusion matrix for 5a predictions

raw_predictions_5a = dnn_clf_5a.predict(input_fn=test_5a_input_fn)

predictions_5a = [p['class_ids'][0] for p in raw_predictions_5a]

confusion_matrix_5a = tf.math.confusion_matrix(y_test, predictions_5a)



with tf.compat.v1.Session():

    con_mat = tf.Tensor.eval(confusion_matrix_5a,feed_dict=None, session=None)
classes=[0,1,2,3,4,5,6,7,8,9]

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

 

con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)
figure = plt.figure(figsize=(6, 5))

sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.coolwarm)

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title('(3) 5 layer model with 20 Neurons')

plt.show()
#Model 5b - 5D with Mixed Neurons each layer (300,200,100,50,25)

start=datetime.now()

feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]

dnn_clf_5b = tf.estimator.DNNClassifier(hidden_units=[300,200,100,50,25], n_classes=10,

                                     feature_columns=feature_cols)



input_fn_5b = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_train}, y=y_train, num_epochs=50, batch_size=50, shuffle=True)

dnn_clf_5b.train(input_fn=input_fn_5b)

time4 = print("Time to calc:", datetime.now()-start)

time4
#evaluate training accuracy (5b)

train_5b_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_train}, y=y_train, shuffle=False)

eval_results_train_5b = dnn_clf_5b.evaluate(input_fn=train_5b_input_fn)

eval_results_train_5b
#evaluate test accuracy (5b)

test_5b_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(

    x={"X": X_test}, y=y_test, shuffle=False)

eval_results_test_5b = dnn_clf_5b.evaluate(input_fn=test_5b_input_fn)

eval_results_test_5b
#confusion matrix for 5b predictions

raw_predictions_5b = dnn_clf_5a.predict(input_fn=test_5b_input_fn)

predictions_5b = [p['class_ids'][0] for p in raw_predictions_5b]

confusion_matrix_5b = tf.math.confusion_matrix(y_test, predictions_5b)



with tf.compat.v1.Session():

    con_mat = tf.Tensor.eval(confusion_matrix_5b,feed_dict=None, session=None)
classes=[0,1,2,3,4,5,6,7,8,9]

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

 

con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)
figure = plt.figure(figsize=(6, 5))

sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.coolwarm)

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title('(4) 5 layer model with mixed neurons')

plt.show()
test_df = pd.read_csv(INPUT/"test.csv")

test_df.head(3)
#Determine number of rows so it can be reshaped

test_df.shape
#format consistently to X_train and X_test

test_df = test_df.astype(np.float32)

test_df = test_df.to_numpy()

test_df = test_df.reshape((28000, 28, 28, 1))

test_df = test_df.astype(np.float32)/ 255.0
#Ensure shape is correct

test_df.shape
sample_df = pd.read_csv(INPUT/"sample_submission.csv")

sample_df.head(3)
test_2a_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"X": test_df}, shuffle=False)

raw_predictions_test_2a = dnn_clf_2a.predict(input_fn=test_2a_input_fn)

predictions_test_2a = [p['class_ids'][0] for p in raw_predictions_test_2a]
submission2a = ()

submission2a = pd.DataFrame({'Imageid' : np.arange(1, len(predictions_test_2a) + 1), 'Label' : predictions_test_2a})

submission2a.head(3)
submission2a.to_csv("1_2a_model.csv", index=False)
test_2b_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"X": test_df}, shuffle=False)

raw_predictions_test_2b = dnn_clf_2b.predict(input_fn=test_2b_input_fn)

predictions_test_2b = [p['class_ids'][0] for p in raw_predictions_test_2b]
submission2b = ()

submission2b = pd.DataFrame({'Imageid' : np.arange(1, len(predictions_test_2b) + 1), 'Label' : predictions_test_2b})

submission2b.head(3)
submission2b.to_csv("2_2b_model.csv", index=False)
test_5a_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"X": test_df}, shuffle=False)

raw_predictions_test_5a = dnn_clf_5a.predict(input_fn=test_5a_input_fn)

predictions_test_5a = [p['class_ids'][0] for p in raw_predictions_test_5a]
submission5a = ()

submission5a = pd.DataFrame({'Imageid' : np.arange(1, len(predictions_test_5a) + 1), 'Label' : predictions_test_5a})

submission5a.head(3)
submission5a.to_csv("3_5a_model.csv", index=False)
test_5b_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"X": test_df}, shuffle=False)

raw_predictions_test_5b = dnn_clf_5b.predict(input_fn=test_5b_input_fn)

predictions_test_5b = [p['class_ids'][0] for p in raw_predictions_test_5b]
submission5b = ()

submission5b = pd.DataFrame({'Imageid' : np.arange(1, len(predictions_test_5b) + 1), 'Label' : predictions_test_5b})

submission5b.head(3)
submission5b.to_csv("4_5b_model.csv", index=False)