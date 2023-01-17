import numpy as np # linear algebra

import pandas as pd # data processing
# load data

df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df = df.drop('customerID', axis=1)
df.shape
df.head()
df.dtypes
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()
df = df.dropna()
df.isnull().sum()
# category explorations

for col in df.columns:

    if df[col].dtype == 'object':

        print(col, df[col].unique())
df['Churn'] = df['Churn'].apply(lambda value: 1 if value == 'Yes' else 0)
for col in df.columns:

    if df[col].dtype == 'object':

        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1).drop(col, axis=1)
df.head()
from xgboost import XGBClassifier
X = df.drop('Churn', axis=1)

y = df['Churn']
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
# scale to handle imbalanced dataset

scale = y_train[y_train == 0].count() / y_train[y_train == 1].count()
xgbmodel = XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=800, scale_pos_weight=scale)
xgbmodel.fit(X_train, y_train)
threshold = 0.6

y_pred_test = xgbmodel.predict_proba(X_test)[:, 1] > threshold

y_pred_train = xgbmodel.predict_proba(X_train)[:, 1] > threshold
print('Train')

print('Precision: {:.2f}% \tRecall: {:.2f}% \t\tF1 Score: {:.2f}%'.format(precision_score(y_train, y_pred_train)*100, recall_score(y_train, y_pred_train)*100, f1_score(y_train, y_pred_train)*100))
print('Test')

print('Precision: {:.2f}% \tRecall: {:.2f}% \t\tF1 Score: {:.2f}%'.format(precision_score(y_test, y_pred_test)*100, recall_score(y_test, y_pred_test)*100, f1_score(y_test, y_pred_test)*100))