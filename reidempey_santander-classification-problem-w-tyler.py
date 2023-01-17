

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')

df_sample = df_train.sample(frac=.3)

y = df_sample['target']
y.value_counts(normalize=True)
df_train.corr()['target'].sort_values(ascending=False)[0:10]
X = df_sample.drop(['target', 'ID_code'], axis=1).copy()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import classification_report, roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)
y_train.value_counts()
param_dictionary = {"n_estimators": [1000]}

clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=3)

# Press Shift-Tab to look at what the arguments are for a function, as well as the defaults for each argument

gs = GridSearchCV(clf, param_dictionary, scoring='roc_auc', n_jobs=1, verbose=2, cv=2)

gs.fit(x_tr, y_tr)

# max depth 5, n estimators 500
#gs.best_index_
#gs.cv_results_
train_predictions = gs.predict(x_tr)

cr = classification_report(y_tr, train_predictions)

roc_auc = roc_auc_score(y_tr, train_predictions)

print("Training Scores:")

print(cr)

print("-"*50)

print("ROC AUC Score: {}".format(roc_auc))
val_predictions = gs.predict(x_val)

cr = classification_report(y_val, val_predictions)

roc_auc = roc_auc_score(y_val, val_predictions)

print('Validation Scores:')

print(cr)

print('-'*50)

print("ROC AUC Score: {}".format(roc_auc))
feat_imports = sorted(list(zip(X_train.columns, gs.best_estimator_.feature_importances_)), key=lambda x:x[1], reverse=True)

feat_imports[0:10]
clf = RandomForestClassifier(n_jobs=-1, max_depth=5, n_estimators=1000, class_weight='balanced', verbose=1)

clf.fit(X_train, y_train)
test_predictions = clf.predict(X_test)

roc_auc = roc_auc_score(y_test, test_predictions)

print('ROC AUC Test Score: {}'.format(roc_auc))
#Error Analysis

probabilities = clf.predict_proba(X_test)
probabilities = probabilities[:,1]

errors = pd.DataFrame()

errors['probs']=probabilities

errors['truth']=y_test.values

errors.head()
errors[errors.truth==1].sort_values(by='probs')[0:10]
X_test.iloc[3995]['var_81']