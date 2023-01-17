# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Iris.csv')
df.head()
df.drop('Id', inplace=True, axis=1)
df.Species.nunique()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df.Species = le.fit_transform(df.Species)
df.Species.sample(3)
#fig, ax = plt.subplots(figsize=(5,5)) 

#sns.set(font_scale=.8)

sns.heatmap(df.corr(),cbar=True,fmt =' .2f', annot=True, cmap='coolwarm')
y = df.Species
df.drop('Species', inplace=True, axis=1)
df.head(5)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(df), columns=df.columns, index= df.index)
X.head(5)
y.head(5)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
import tensorflow as tf
SepalLengthCm =tf.feature_column.numeric_column("SepalLengthCm")

SepalWidthCm =tf.feature_column.numeric_column("SepalWidthCm")

PetalLengthCm =tf.feature_column.numeric_column("PetalLengthCm")

PetalWidthCm =tf.feature_column.numeric_column("PetalWidthCm")
feature_columns = [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
cols = ["SepalLengthCm","SepalWidthCm","PetalLengthCm", "PetalWidthCm"]
clasfr = tf.estimator.LinearClassifier(feature_columns, n_classes=3)
train_input_fn = tf.estimator.inputs.pandas_input_fn(X_train, y=y_train, num_epochs=5000, shuffle=True)
clasfr.train(input_fn=train_input_fn)
#eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, num_epochs=1, shuffle=False)
#clasfr.evaluate(input_fn=eval_input_fn)
test_input_fn = tf.estimator.inputs.pandas_input_fn(X_test, num_epochs=1, shuffle=False)
result =list(clasfr.predict(input_fn=test_input_fn))
y_pred = []

for each in result:

    y_pred.append(each['class_ids'][0])
from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test,y_pred))
cols_n = ['PetalLengthCm', 'SepalWidthCm']

X_train_n = X_train[cols_n]

X_test_n = X_test[cols_n]
clasfr = tf.estimator.LinearClassifier([PetalLengthCm,SepalWidthCm] , n_classes=3)
train_input_fn = tf.estimator.inputs.pandas_input_fn(X_train_n, y=y_train, num_epochs=5000, shuffle=True)
clasfr.train(input_fn=train_input_fn)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test_n, y=y_test, num_epochs=1, shuffle=False)
clasfr.evaluate(input_fn=eval_input_fn)