# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bank = pd.read_csv('/kaggle/input/bank-notes/bank_note_data.csv')
bank.head()
bank.info()
sns.countplot(x='Class', data=bank)
sns.pairplot(bank, hue='Class')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(bank.drop('Class', axis=1))
scaled_features = scaler.transform(bank.drop('Class', axis=1))
df_feat = pd.DataFrame(scaled_features, columns = bank.columns[:-1])

df_feat.head()
X = df_feat

y = bank['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
import tensorflow as tf
df_feat.columns
feat_cols = []



for col in df_feat.columns:

    feat_cols.append(tf.feature_column.numeric_column(col))
feat_cols
classifier = tf.estimator.DNNClassifier(hidden_units=[10,20,10], n_classes=2, feature_columns=feat_cols)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=20, shuffle=True)
classifier.train(input_fn=input_func, steps = 500)
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
note_predictions = list(classifier.predict(input_fn=pred_fn))
note_predictions[0]
final_preds = []



for pred in note_predictions:

    final_preds.append(pred['class_ids'][0])
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test,final_preds))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_preds =rfc.predict(X_test)
print(classification_report(y_test,rfc_preds))
print(confusion_matrix(y_test,rfc_preds))