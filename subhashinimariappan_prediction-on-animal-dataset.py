#importing libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
#importing data from my workshop

file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter04/Dataset/openml_phpZNNasq.csv'
#reading data

df = pd.read_csv(file_url)
#data preprocessing

df.drop(columns='animal', inplace=True)

y = df.pop('type')
#spiltting test , train data.

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.4, random_state=188)
#applying random forest classifier

rf_model = RandomForestClassifier(random_state=42, n_estimators=1)

rf_model.fit(X_train, y_train)
#prediction on model

train_preds = rf_model.predict(X_train)

test_preds = rf_model.predict(X_test)
#accuracy on model

train_acc = accuracy_score(y_train, train_preds)

test_acc = accuracy_score(y_test, test_preds)
#printing for accuracy

print(train_acc)

print(test_acc)
rf_model2 = RandomForestClassifier(random_state=42, n_estimators=30)

rf_model2.fit(X_train, y_train)
train_preds2 = rf_model2.predict(X_train)

test_preds2 = rf_model2.predict(X_test)
train_acc2 = accuracy_score(y_train, train_preds2)

test_acc2 = accuracy_score(y_test, test_preds2)
print(train_acc2)

print(test_acc2)