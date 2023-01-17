import pandas
import sklearn
import numpy as np
import os
cwd = os.getcwd()
train_adult = pandas.read_csv("../input/adult-pmr3508/train_data.csv" ,names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        skiprows = 1,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test_adult = pandas.read_csv("../input/adult-pmr3508/test_data.csv" ,names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        skiprows = 1,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
natrain_adult = train_adult.dropna()
natest_adult = test_adult.dropna()
import matplotlib.pyplot as plt
natest_adult["Capital Gain"].value_counts().plot(kind='bar')
natrain_adult["Hours per week"].value_counts().plot(kind='pie')
targetxrace = pandas.crosstab(natrain_adult["Race"],natrain_adult["Target"],margins=False)
targetxrace.plot(kind='bar',stacked=False)
targetxrace
targetxrace = pandas.crosstab(natrain_adult["Sex"],natrain_adult["Target"],margins=False)
targetxrace.plot(kind='bar',stacked=True)
from sklearn import preprocessing
adult_train = natrain_adult[["Age", "Workclass", "Education-Num", "Occupation", "Race", "Sex", "Capital Gain", 
                               "Capital Loss","Hours per week"]]
adult_test = natest_adult[["Age", "Workclass", "Education-Num", "Occupation", "Race", "Sex", "Capital Gain", 
                               "Capital Loss","Hours per week"]]
Xtrainadult = adult_train.apply(preprocessing.LabelEncoder().fit_transform)
Ytrainadult = natrain_adult.Target
Xtestadult = adult_test.apply(preprocessing.LabelEncoder().fit_transform)
from sklearn.neighbors import KNeighborsClassifier as knnClassifier
knn = knnClassifier(n_neighbors = 32)
from sklearn.neighbors import KNeighborsClassifier as knnClassifier
from sklearn.model_selection import cross_val_score
results=[0]*30
for i in range(30):
    knn = knnClassifier(n_neighbors = 32+i)
    scores = cross_val_score(knn, Xtrainadult, Ytrainadult,cv=10)
    results[i] = np.mean(scores)
results
knn = knnClassifier(n_neighbors = 32)
scores = cross_val_score(knn, Xtrainadult, Ytrainadult, cv=10)
np.mean(scores)
knn.fit(Xtrainadult, Ytrainadult )
predYtest = knn.predict(Xtestadult)
income = pandas.DataFrame(predYtest)
income.to_csv("submission.csv",header = ["income"], index_label = "Id")