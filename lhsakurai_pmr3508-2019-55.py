import pandas as pd

import sklearn

import matplotlib.pyplot as plt
import os

cwd = os.getcwd()

cwd
os.listdir('../input')
train_data = "../input/ucidata/adult_data.csv"

test_data = "../input/ucidata/adult_test.csv"
adult = pd.read_csv(train_data, skipinitialspace = True, na_values = "?")
adult.shape
adult.head()
list(adult)
adult["target"].value_counts().plot(kind="bar")
adult["relationship"].value_counts().plot(kind="bar")
adult["marital_status"].value_counts().plot(kind="bar")
adult["age"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["education_num"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
adult["hours_per_week"].value_counts().plot(kind="bar")
nadult = adult.dropna()
nadult.shape
test_adult = pd.read_csv(test_data, skipinitialspace = True)
test_adult.shape
ntest_adult = test_adult
ntest_adult.shape
from sklearn import preprocessing
attributes = ["age", "workclass", "education_num", "marital_status", "relationship", "capital_gain", "capital_loss", "hours_per_week"]
train_adult_x = nadult[attributes].apply(preprocessing.LabelEncoder().fit_transform)
train_adult_y = nadult.target
test_adult_x = test_adult[attributes].apply(preprocessing.LabelEncoder().fit_transform)
test_adult_y = test_adult.target
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
# columns = ['neighbors', 'cross', 'accuracy']

# results = [columns]

# for n in range (30, 60):

#     neighbors = n

#     cross = 10

#     knn = KNeighborsClassifier(n_neighbors = neighbors)

#     scores = cross_val_score(knn, train_adult_x, train_adult_y, cv = cross)

# #     scores

#     knn.fit(train_adult_x, train_adult_y)

#     test_pred_y = knn.predict(test_adult_x)

# #     test_pred_y

#     acc = accuracy_score(test_adult_y, test_pred_y)

#     results.append([neighbors,cross, acc])
# results
neighbors = 59

cross = 10

knn = KNeighborsClassifier(n_neighbors = neighbors)

scores = cross_val_score(knn, train_adult_x, train_adult_y, cv = cross)

scores
knn.fit(train_adult_x, train_adult_y)

test_pred_y = knn.predict(test_adult_x)

test_pred_y
accuracy_score(test_adult_y, test_pred_y)
kaggle_data = '../input/kaggle-data/test_data.csv'

kaggle_adult = pd.read_csv(kaggle_data)
kaggle_adult_x = kaggle_adult[attributes].apply(preprocessing.LabelEncoder().fit_transform)

kaggle_pred_y = knn.predict(kaggle_adult_x)

kaggle_pred_y
len(kaggle_pred_y)
id_index = pd.DataFrame({'Id' : list(range(len(kaggle_pred_y)))})

income = pd.DataFrame({'income' : kaggle_pred_y})
result = id_index.join(income)

result
result.to_csv("submission.csv", index = False)