import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data
profile = ProfileReport(data)
profile.to_notebook_iframe()
y = data['DEATH_EVENT']

X = data.drop('DEATH_EVENT', axis=1)
scaler = MinMaxScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



log_acc = model.score(X_test, y_test)

log_f1 = f1_score(y_test, y_pred)
model = SVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



svm_acc = model.score(X_test, y_test)

svm_f1 = f1_score(y_test, y_pred)
model = MLPClassifier(hidden_layer_sizes=(128, 128))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



nn_acc = model.score(X_test, y_test)

nn_f1 = f1_score(y_test, y_pred)
print(f"Logistic Regression:\nAccuracy: {log_acc}\nF1 Score: {log_f1}\n")

print(f"Support Vector Machine:\nAccuracy: {svm_acc}\nF1 Score: {svm_f1}\n")

print(f"Neural Network:\nAccuracy: {nn_acc}\nF1 Score: {nn_f1}\n")