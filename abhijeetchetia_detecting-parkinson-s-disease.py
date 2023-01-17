import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split as tts

from sklearn.model_selection import KFold, GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
# Importing the dataset



file = "../input/parkinsons-disease-data/parkinsons.data"

data = pd.read_csv(file)
data.head(5)
print("The columns in the dataset are as follows:- \n", data.columns.tolist())

print("\nNumber of missing values in the dataset: ", data.isnull().sum().sum())

print("\nUnique values in 'name' and 'status' column:- \n", data[['name', 'status']].nunique())

print("\nShape of the dataset: ", data.shape)
y = data['status'] # Labels

predictors = data.drop(['status', 'name'], axis=1)
y.value_counts()
scale = MinMaxScaler((-1, 1))

X = scale.fit_transform(predictors.values)
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=7)
# Building Model



model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X, y)
accuracy_score(y_test, y_pred)
model.fit(X, y)