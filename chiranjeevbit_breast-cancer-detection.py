import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import requests
# Importing and cleaning data using pandas library

data = pd.read_csv('../input/breast-cancer.csv')

del data['Unnamed: 32']
## Observe the data

data.head()
data.describe()
# Let's convert "M to 0,  F to 1 values

data.diagnosis = [ 1 if each == "B" else 0 for each in data.diagnosis]





data.head()
# ## We have left first two columns and taken other columns as input features

# X = data.iloc[:, 2:].values



# # 2nd column is output labels

# y = data.iloc[:, 1].values





# OR 



# Let's determine the values of y and x axes

y = data.diagnosis.values

X = data.drop(["diagnosis"], axis=1)
print(X.shape)

print(y.shape)
# Now we are doing normalization. 

# Normalization is important 

# Because if some of our columns have very high values, they will suppress other columns and do not show much.

# Formule used  : (x- min(x)) / (max(x) - min(x))

x = (X - np.min(X)) / (np.max(X) - np.min(X)).values

x.head()
# Now we reserve 80% of the values as 'train' and 10% as 'test'.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state=42)





print("x_train :", x_train.shape)

print("x_test :", x_test.shape)

print("y_train :", y_train.shape)

print("y_test :", y_test.shape)
import tensorflow as tf

import tensorflow.contrib.learn.python



from tensorflow.contrib.learn.python import learn as learn


#classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]

classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=10)

tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")
classifier.fit(x_train, y_train, steps=200, batch_size=20)

note_predictions = classifier.predict(x_test)



from sklearn.metrics import confusion_matrix,classification_report

lst = list(note_predictions)
print(confusion_matrix(y_test,lst))

print(classification_report(y_test,lst))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(x_train,y_train)
rfc_preds = rfc.predict(x_test)

print(classification_report(y_test,rfc_preds))
rfc_score_train = rfc.score(x_train, y_train)

print("Training score: ",rfc_score_train)

rfc_score_test = rfc.score(x_test, y_test)

print("Testing score: ",rfc_score_test)
from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(x_train, y_train)

logis_score_train = logis.score(x_train, y_train)

print("Training score: ",logis_score_train)

logis_score_test = logis.score(x_test, y_test)

print("Testing score: ",logis_score_test)

#decision tree

from sklearn.ensemble import RandomForestClassifier

dt = RandomForestClassifier()

dt.fit(x_train, y_train)

dt_score_train = dt.score(x_train, y_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(x_test, y_test)

print("Testing score: ",dt_score_test)

#Model comparison

models = pd.DataFrame({

        'Model'          : ['Logistic Regression',  'Decision Tree', 'Random Forest'],

        'Training_Score' : [logis_score_train,  dt_score_train, rfc_score_train],

        'Testing_Score'  : [logis_score_test, dt_score_test, rfc_score_test]

    })

models.sort_values(by='Testing_Score', ascending=False)
