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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/anomaly-detection/creditcard.csv')
df.head(-1)['Class']
df.isnull().sum()
df.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(df.drop('Class',axis=1))

df_feat = pd.DataFrame(data=scaled,columns=df.columns[:-1])
df_feat.head()
X=df_feat
y=df['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

import tensorflow as tf
feat_cols = []
for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
feat_cols
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=200,num_epochs=80,shuffle=True)
classifier = tf.compat.v1.estimator.DNNClassifier(hidden_units=[30,40,40,40,40,30],n_classes=2,feature_columns=feat_cols)
classifier.train(input_fn=input_func,steps=200)
pred_fn= tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
predictions = list(classifier.predict(input_fn=pred_fn))
final_preds=[]
for pred in predictions:
    final_preds.append(pred['class_ids'][0])
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,final_preds))
print(confusion_matrix(y_test,final_preds))
print(accuracy_score(y_test,final_preds))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=400)
rf.fit(X_train,y_train)
preds = rf.predict(X_test)
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))
print(accuracy_score(y_test,preds))
