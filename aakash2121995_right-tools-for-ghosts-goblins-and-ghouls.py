# Import the libraries

import pandas as pd

import re

import tensorflow as tf

from tensorflow.contrib import learn

import warnings

warnings.filterwarnings('ignore')
# Library for tensorflow logging

import logging

logging.getLogger().setLevel(logging.INFO)

df = pd.read_csv("../input/train.csv")
# Selecting features

features = ["bone_length","rotting_flesh","hair_length","color","has_soul"]

X = df[features]

y = df["type"]
# Encoding type (Ghost,Ghouls,Goblin) and color

from sklearn.preprocessing import LabelEncoder as LE

letype = LE()

y = letype.fit_transform(y)

lecolor = LE()

X["color"] = lecolor.fit_transform(X["color"])
# splitting function used for cross validation

from sklearn.cross_validation import train_test_split

# current test size = 0 to permit the usage of whole training data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.0, random_state=9)
# define a network with a single hidden RELU layer of 15 hidden units 

tf_clf_dnn = learn.DNNClassifier(hidden_units=[15], n_classes=3)

tf_clf_dnn.fit(X_train, y_train,max_steps=5500)

from sklearn.metrics import accuracy_score as as_

# print(as_(y_test,tf_clf_dnn.predict(X_test)))

print(as_(y_train,tf_clf_dnn.predict(X_train)))
# Reading csv into test_df

test_df = pd.read_csv("../input/test.csv")

X_test = test_df[features]



# Reading ID

id_ = test_df["id"]



# Encoding color

X_test["color"] = lecolor.transform(X_test["color"])

 

# Prediction and Decoding into labels

pred = tf_clf_dnn.predict(X_test)

pred = letype.inverse_transform(pred)

output = pd.DataFrame({"id": id_,"type":pred})

output.to_csv('ghostPred.csv',index=False)