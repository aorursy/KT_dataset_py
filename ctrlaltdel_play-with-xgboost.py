import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
# load data

dataset = pd.read_csv('../input/diabetes.csv')
# convert column names to lower case

dataset.columns = [item.lower() for item in dataset.columns]
# extract data and labels

data_columns = [column for column in dataset.columns if column != 'outcome']

label_column = list(set(dataset.columns) - set(data_columns))



data = dataset[data_columns]

labels = dataset[label_column]
# split into train and test sets

seed = 7

test_size = 0.33



X_train, X_test, y_train, y_test = train_test_split(data.values, labels.values, test_size = test_size,

                                                    random_state = seed, stratify= labels)
# fit model on training data

y_train = y_train.ravel() # transform to accepted dimensions

y_test = y_test.ravel()

model = XGBClassifier()

model.fit(X_train, y_train)



# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

 
# xgboost can report on performance during training, that is, after each tree is added

eval_set = [(X_test, y_test)] # establish eval set

# train with verbose = true

model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
# fit model on training data with early stopping using the early_stopping_rounds parameter

model = XGBClassifier()

eval_set = [(X_test, y_test)]

model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", 

          eval_set=eval_set, verbose=True)



# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))