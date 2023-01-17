import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_auc_score, classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
import tensorflow as tf
diabetes_data = pd.read_csv('../input/diabetes.csv')
diabetes_data.columns
diabetes_data.head()
sns.pairplot(data=diabetes_data, hue='Outcome')
print(diabetes_data['Outcome'].value_counts())
diabetes_data.info()
scaler = StandardScaler()
scaler.fit(diabetes_data.iloc[:,:-1])
scaled_features = scaler.transform(diabetes_data.iloc[:,:-1])
df_feat = pd.DataFrame(scaled_features,columns=diabetes_data.columns[:-1])
df_feat.head()
X = df_feat

y = diabetes_data.iloc[:,-1]

replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']



for column in replace_zero:

    X[column] = X[column].replace(0, np.NaN)

    mean = int(X[column].mean(skipna=True))

    X[column] = X[column].replace(np.NaN, mean)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=101)
feat_cols = []



for col in X.columns:

    feat_cols.append(tf.feature_column.numeric_column(col))
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True)
classifier = tf.estimator.DNNClassifier(hidden_units = [10,20,10],n_classes=2,feature_columns=feat_cols)
classifier.train(input_fn=input_func, steps=50)
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
note_predictions = list(classifier.predict(input_fn=pred_fn))
final_preds  = []

for pred in note_predictions:

    final_preds.append(pred['class_ids'][0])
print(classification_report(y_test,final_preds))

print(confusion_matrix(y_test,final_preds))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test,predictions))
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test,predictions))