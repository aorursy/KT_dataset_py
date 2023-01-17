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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data = pd.read_csv('../input/Iris.csv')
# Let's do some data explatory first
data.head()
# We don't need Id column
data.drop('Id', axis=1, inplace=True)
sns.pairplot(data, hue='Species')
# let's turn 'speciaes' into a numeric values becuase this what we need for classification
data["Species"] = data["Species"].map({"Iris-setosa":0,"Iris-virginica":1,"Iris-versicolor":2})
data.info()
import tensorflow as tf
from sklearn.model_selection import train_test_split
y = data['Species']
X= data.drop('Species', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
feat_cols = [tf.feature_column.numeric_column(col) for col in X.columns]
feat_cols[:3]
# First way using tf.estimator.inputs. NOTE: you have to provide shuffle argument 
# otherwise you will get an error. Since we used trains test split I already shuffled it
input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=20, 
                                                 num_epochs=5, shuffle=False)

# # second way: build your own function
# def input_fn(df,labels):
#     feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in X.columns}
#     label = tf.constant(labels.values, shape = [labels.size,1])
#     return feature_cols,label
#We added 3 hidden layer with 10 20 and 10 nodes respectively. 
# We are overkilling for such a simple problem. You can test different hiddent unites and nodes 
classifier = tf.estimator.DNNClassifier(hidden_units=[10,10], n_classes=3, 
                                        feature_columns=feat_cols)
# Train the estimator run the following if you used tf.estimator.inputs
classifier.train(input_fn=input_fn, steps=50)

# # if you build your own input funtion above, you will need to run the following
# classifier.train(input_fn=lambda:input_fn(X_train, y_train), steps=50)
# In order to evaluate model I need to  creatr another input function for evaluation
# I will use tf.estimator.inputs.pandas_input_fn
eval_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=len(X_test), 
                                              shuffle=False)

# # if you want to use input_fn you built, you will need to run the following
# # this time pass X_test, y_test to you input function
# classifier.evaluate(input_fn=lambda:input_fn(X_test, y_test), steps=50)
eval_result = classifier.evaluate(input_fn=eval_fn)
print(eval_result)
# we will create another input function for predictions and call it pred_fn
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
# classifier.predict is a generator, we need cast result to a list
predictions = list(classifier.predict(input_fn=pred_fn))
predictions[:3]
# value for the 'class ids' is the selected class
# let's collect them in a list
final_pred = [pred['class_ids'][0] for pred in predictions]
final_pred[:10]
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, final_pred))
print(classification_report(y_test, final_pred))
# it seems it is doing pretty good
# question can we do as good as NN with a simpler algorithms such as decision tree or 
# logistic regression. i will answer this in the second part but for now that is it
