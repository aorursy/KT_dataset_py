import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
TARGET = "is_female"
df = pd.read_csv("../input/train.csv",low_memory=False)
print(df.shape)
df.head()
df2 = df.dropna(axis=1)
print (df2.shape)
# to get all columns names
df2.columns

X= df2.drop(['train_id','is_female'], axis=1)
Y = df2.is_female
test = pd.read_csv("../input/test.csv",low_memory=False)
print(test.shape)
# keep those columns from training data X
test3 = test.reindex(columns=X.columns, fill_value=0)
test3.shape

print (Y.shape, X.shape, test3.shape)
X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=0.33, random_state=42)
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
Ntree = 500
clf = RandomForestClassifier(n_estimators=Ntree,random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
y_prob = clf.predict_proba(X_test)
from sklearn import metrics
# let check which score do we need,   we need column 1  not column 0
y_prob
metrics.roc_auc_score(y_test, y_prob[:,1])
Ntree = 500
clf2 = RandomForestClassifier(n_estimators=Ntree,random_state=1234)
clf2.fit(X, Y)
y_submit = clf2.predict_proba(test3)[:,1]
test['is_female'] = y_submit
ans = test[['test_id','is_female']]
ans.to_csv('submit.csv', index=None)
