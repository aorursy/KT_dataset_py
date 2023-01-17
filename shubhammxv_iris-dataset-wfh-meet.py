import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RandomFC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data_input = pd.read_csv('../input/iris/Iris.csv')
data_input.head()
data_input.shape
X = data_input.drop(['Id', 'Species'], axis=1)
y = data_input['Species']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
random_forest = RandomFC(n_estimators=10)
random_forest.fit(X_train,y_train)
y_pred = random_forest.predict(X_test)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
accuracy_score(y_test,y_pred)
input_train = pd.read_csv('../input/wfh-ds/train_data.csv')
input_test = pd.read_csv('../input/wfh-ds/test_data.csv', header=None)
X1 = input_train.drop(['Class'], axis=1)
y1 = input_train['Class']
X1_train, X1_test, y1_train, y1_test = train_test_split( X1, y1, test_size=0.2)
random_forest = RandomFC(n_estimators=100)
random_forest.fit(X1_train,y1_train)
y1_pred = random_forest.predict(X1_test)
confusion_matrix(y1_test, y1_pred)
accuracy_score(y1_test,y1_pred)
y2_pred = random_forest.predict(input_test)
y2_pred