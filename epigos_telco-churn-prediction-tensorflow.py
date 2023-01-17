import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
import tensorflow as tf

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.info()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['SeniorCitizen'].unique()
df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
df['SeniorCitizen'].unique()
df['TotalCharges'].isnull().sum()
df['TotalCharges'].fillna(value=df['tenure'] * df['MonthlyCharges'], inplace=True)
def churn_to_numeric(value):
    if value.lower() == 'yes':
        return 1
    return 0
df['Churn'] = df['Churn'].apply(churn_to_numeric)
df.describe()
df.info()
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
tenure = tf.feature_column.numeric_column('tenure')
monthly_charges = tf.feature_column.numeric_column('MonthlyCharges')
total_charges = tf.feature_column.numeric_column('TotalCharges')
col_unique_val_counts = []
cat_columns = []
for col in X.columns:
    if X[col].dtype.name != 'object':
        continue
    unique_vals = X[col].unique()
    col_unique_val_counts.append(len(unique_vals))
    cat_columns.append(col)
    print(col, "->",unique_vals)
cat_cols = [tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=size) 
            for col, size in zip(cat_columns, col_unique_val_counts)]
num_cols = [tenure, monthly_charges, total_charges]
feature_columns = num_cols + cat_cols
n_classes = 2 # churn Yes or No
batch_size = 100
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=batch_size,num_epochs=1000, shuffle=True)
linear_model= tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=n_classes)
linear_model.train(input_fn=input_func, steps=10000) # uinsg 10k steps
eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=batch_size,
      num_epochs=1,
      shuffle=False)
linear_model.evaluate(eval_input_func)
pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=batch_size,
      num_epochs=1,
      shuffle=False)
preds = linear_model.predict(pred_input_func)
predictions = [p['class_ids'][0] for p in preds]
from sklearn.metrics import classification_report
target_names = ['No', 'Yes']
print(classification_report(y_test, predictions, target_names=target_names))
