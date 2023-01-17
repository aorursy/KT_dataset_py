import numpy as np

import pandas as pd
data = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent.csv')
print(data.shape)

data.head(10)
data.drop(data.columns[0], axis=1, inplace=True)
data['floor'].replace(to_replace='-', value=0, inplace=True)
data['animal'].replace(to_replace='not acept', value=0, inplace=True)

data['animal'].replace(to_replace='acept', value=1, inplace=True)
data['furniture'].replace(to_replace='not furnished', value=0, inplace=True)

data['furniture'].replace(to_replace='furnished', value=1, inplace=True)
for col in ['hoa', 'rent amount', 'property tax', 'fire insurance', 'total']:

    data[col].replace(to_replace='R\$', value='', regex=True, inplace=True)

    data[col].replace(to_replace=',', value='', regex=True, inplace=True)
data['hoa'].replace(to_replace='Sem info', value='0', inplace=True)
data['hoa'].replace(to_replace='Incluso', value='0', inplace=True)

data['property tax'].replace(to_replace='Incluso', value='0', inplace=True)
data = data.astype(dtype=np.int64)
data.isin(['Incluso']).any()
data = data.sample(frac=1).reset_index(drop=True)
y = data['city']

X = data.drop('city', axis=1)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X)

X = scaler.transform(X)
pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
log_model = LogisticRegression(penalty='l2', verbose=1)

svm_model = SVC(kernel='rbf', verbose=1)

nn_model = MLPClassifier(hidden_layer_sizes=(16, 16), activation='relu', solver='adam', verbose=1)
log_model.fit(X_train, y_train)

svm_model.fit(X_train, y_train)

nn_model.fit(X_train, y_train)
print(log_model.score(X_test, y_test))

print(svm_model.score(X_test, y_test))

print(nn_model.score(X_test, y_test))
from sklearn.metrics import f1_score
log_pred = log_model.predict(X_test)

svm_pred = svm_model.predict(X_test)

nn_pred = nn_model.predict(X_test)
print(f1_score(log_pred, y_test))

print(f1_score(svm_pred, y_test))

print(f1_score(nn_pred, y_test))