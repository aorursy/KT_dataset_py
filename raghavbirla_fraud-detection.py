import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import tensorflow as tf
import shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
%matplotlib inline
df = pd.read_csv("../input/creditcard.csv")
df.head()
df.describe()
df.shape
df.dtypes
df['Time']
sns.countplot(x=df['Class'])
df.isnull().any()
X = df.drop(['Time','Class'],axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=324)
dtclassifier = DecisionTreeClassifier(max_leaf_nodes=20,random_state=0)
dtclassifier.fit(X_train,y_train)
prediction = dtclassifier.predict(X_test)
prediction[:10]
y_test[:10]
accuracy_score(y_true=y_test, y_pred=prediction)
lrclassifier = LogisticRegression()
lrclassifier.fit(X_train,y_train)
prediction = lrclassifier.predict(X_test)
prediction[:10]
y_test[:10]
accuracy_score(y_true=y_test, y_pred=prediction)
X.columns
features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
label = ['Class']
print(features)
print(label)
def make_input_fn(num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    batch_size=128,
    shuffle=True,
    num_epochs = num_epochs,
    queue_capacity=1000,
    num_threads=1)
def make_evaluation_input_fn(num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
    x=X_test,
    y=y_test,
    batch_size=128,
    num_epochs=num_epochs,
    shuffle=True,
    queue_capacity=1000,
    num_threads=1)
def make_prediction_input_fn(num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
    x=X_test,
    y=None,
    batch_size=128,
    num_epochs=num_epochs,
    shuffle=True,
    queue_capacity=1000,
    num_threads=1)
def make_feature_cols():
    input_cols = [tf.feature_column.numeric_column(k) for k in features]
    return input_cols
tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR='credit card fraud detection'
shutil.rmtree(OUTDIR, ignore_errors=True) # start fresh each time

LinearClassifierModel = tf.estimator.LinearClassifier(feature_columns=make_feature_cols(), model_dir=OUTDIR)
LinearClassifierModel.train(input_fn=make_input_fn(num_epochs = 10))
LinearClassifierModel.evaluate(input_fn=make_evaluation_input_fn(num_epochs=10))
tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors=True) # start fresh each time
DNNClassifierModel = tf.estimator.DNNClassifier(hidden_units=[32,8,2], feature_columns=make_feature_cols(),
                                                model_dir=OUTDIR, activation_fn=tf.nn.relu,
                                                n_classes=2, optimizer='Adam', dropout=0.2,)
DNNClassifierModel.train(input_fn=make_input_fn(num_epochs=10))
DNNClassifierModel.evaluate(input_fn=make_evaluation_input_fn(num_epochs=10))
