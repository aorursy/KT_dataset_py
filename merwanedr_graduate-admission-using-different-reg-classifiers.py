from sklearn import svm
from sklearn import linear_model
import pandas as pd
import math
data = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
data.head(6)
def preprocess(data):
    X = data.iloc[:, 1:8]  # all rows, all the grades and no labels
    y = data.iloc[:, 8]  # all rows, Chance of admit only

    return X, y
from sklearn.model_selection import train_test_split

all_X, all_y = preprocess(data)
X_train, X_test, y_train, y_test = train_test_split(all_X, all_y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]
for item in classifiers:
    print(item)
    clf = item
    clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# Predict on new data
clf.predict(X_test[15:25])
# Real answer
y_test[15:25]
