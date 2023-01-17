"""
Data source: 
Mohan S Acharya, Asfia Armaan, Aneeta S Antony : 
A Comparison of Regression Models for Prediction of Graduate Admissions, 
IEEE International Conference on Computational Intelligence in Data Science 2019
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
data_file = "../input/graduate-admissions/Admission_Predict_Ver1.1.csv"
data = pd.read_csv(data_file)
# Column keys
SER_KEY = "Serial No."
GRE_KEY = "GRE Score"
TOEFL_KEY = "TOEFL Score"
UNI_KEY = "University Rating"
SOP_KEY = "SOP"
LOR_KEY = "LOR " # Note the trailing space character
CGPA_KEY = "CGPA"
RES_KEY = "Research"
CHANCE_ADMIT_KEY = "Chance of Admit"
data
data.hist(column=GRE_KEY, bins=40)
print(f"Mean GRE score: {data[GRE_KEY].mean()}")
print(f"Median GRE score: {data[GRE_KEY].median()}")
print(f"Min GRE score: {data[GRE_KEY].min()}")
print(f"Max GRE score: {data[GRE_KEY].max()}")
data.hist(column=TOEFL_KEY, bins=20)
print(f"Mean TOEFL score: {data[TOEFL_KEY].mean()}")
print(f"Median TOEFL score: {data[TOEFL_KEY].median()}")
print(f"Min TOEFL score: {data[TOEFL_KEY].min()}")
print(f"Max TOEFL score: {data[TOEFL_KEY].max()}")
data.hist(column=UNI_KEY, bins=5)
print(f"Mean University rating: {data[UNI_KEY].mean()}")
print(f"Median University rating: {data[UNI_KEY].median()}")
print(f"Min University rating: {data[UNI_KEY].min()}")
print(f"Max University rating: {data[UNI_KEY].max()}")
data.hist(column=SOP_KEY, bins=10)
print(f"Mean SOP: {data[SOP_KEY].mean()}")
print(f"Median SOP: {data[SOP_KEY].median()}")
print(f"Min SOP: {data[SOP_KEY].min()}")
print(f"Max SOP: {data[SOP_KEY].max()}")
data.hist(column=LOR_KEY, bins=10)
print(f"Mean LOR: {data[LOR_KEY].mean()}")
print(f"Median LOR: {data[LOR_KEY].median()}")
print(f"Min LOR: {data[LOR_KEY].min()}")
print(f"Max LOR: {data[LOR_KEY].max()}")
data.hist(column=CGPA_KEY, bins=10)
print(f"Mean CGPA: {data[CGPA_KEY].mean()}")
print(f"Median CGPA: {data[CGPA_KEY].median()}")
print(f"Min CGPA: {data[CGPA_KEY].min()}")
print(f"Max CGPA: {data[CGPA_KEY].max()}")
# Separate the data into training and validation sets
train_full = data.iloc[:400, :]
test_full = data.iloc[400:500, :]

# The last column, chance of admit, is how the accuracy of a model will be measured
train_X = train_full.iloc[:, 1:8]
train_y = train_full.iloc[:, 8]

test_X = test_full.iloc[:, 1:8]
test_y = test_full.iloc[:, 8]
# Standardized version of data
std_train_X = (train_X - train_X.mean()) / train_X.std()
std_test_X = (test_X - test_X.mean()) / test_X.std()
# Dimensionality reduction
pca = PCA(n_components=3)
pca.fit_transform(train_X)
pca.explained_variance_ratio_
abs(pca.components_)
def run_classifiers(classifiers, train_X, train_y, test_X, test_y):
    """
    Fits each classifier to the training data and runs it on the test data.
    Prints out the training and test mean squared errors. 
    """
    # Baseline: a random predictor that assigns a probability of 0.5 to each data point
    rand_train_mse = mean_squared_error(train_y.values, np.full(400, 0.5))
    rand_test_mse = mean_squared_error(test_y.values, np.full(100, 0.5))
    print(f"Random training MSE = {rand_train_mse}")
    print(f"Random test MSE = {rand_test_mse}")
    print()
    
    for clf in classifiers:
        print(f"Classifier: {clf}")
        clf.fit(train_X, train_y)
        train_mse = mean_squared_error(train_y.values, clf.predict(train_X))
        test_mse = mean_squared_error(test_y.values, clf.predict(test_X))

        print(f"Training MSE = {train_mse}")
        print(f"Validation MSE = {test_mse}")
        print()

classifiers = [
    LinearRegression(),
    BayesianRidge(),
    SVR(gamma="scale")
]
run_classifiers(classifiers, train_X, train_y, test_X, test_y)
