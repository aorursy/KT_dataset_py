import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv('../input/HR_comma_sep.csv')
data.head()
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])

data['sales'] = le.fit_transform(data['sales'])
data.head()
X = data.drop(['left'], axis=1)

y = data['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)
accuracy_score(y_test, rf_preds)
print(classification_report(y_test, rf_preds))
%matplotlib inline

feats = {} # a dict to hold feature_name: feature_importance

for feature, importance in zip(data.columns, rf_clf.feature_importances_):

    feats[feature] = importance #add the name/value pair 



importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)