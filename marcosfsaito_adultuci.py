import pandas as pd
import sklearn
import numpy as np
import os
cwd = os.getcwd()
train_adult = pd.read_csv("../input/adultb/train_data.csv" ,names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        skiprows = 1,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test_adult = pd.read_csv("../input/adultb/test_data.csv" ,names=[
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
targetxrace = pd.crosstab(natrain_adult["Race"],natrain_adult["Target"],margins=False)
targetxrace.plot(kind='bar',stacked=False)
targetxrace
targetxrace = pd.crosstab(natrain_adult["Sex"],natrain_adult["Target"],margins=False)
targetxrace.plot(kind='bar',stacked=True)
from sklearn import preprocessing
adult_train = natrain_adult[["Age", "Workclass", "Education-Num", "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]
adult_test = natest_adult[["Age", "Workclass", "Education-Num", "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]
Xtrainadult = adult_train.apply(preprocessing.LabelEncoder().fit_transform)
Ytrainadult = natrain_adult.Target
Xtestadult = adult_test.apply(preprocessing.LabelEncoder().fit_transform)
from sklearn.neighbors import KNeighborsClassifier as knnClassifier
knn = knnClassifier(n_neighbors = 35)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xtrainadult, Ytrainadult,cv=10)
scores
knn.fit(Xtrainadult, Ytrainadult)
predYtest = knn.predict(Xtestadult)
predYtest
income = pd.DataFrame(predYtest)
income.to_csv("submission.csv",header = ["income"], index_label = "Id")
train = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv")
test = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")
ntrain = train.dropna()
from sklearn import preprocessing
Xtrain = train.iloc[:,0:-1]
Ytrain = train.Target
Xtest = test
nXtrain = Xtrain.apply(preprocessing.LabelEncoder().fit_transform)
nXtest = Xtest.apply(preprocessing.LabelEncoder().fit_transform)
from sklearn.neighbors import KNeighborsClassifier as KnnC
knn = KnnC(n_neighbors = 30)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, nXtrain, Ytrain, cv=10)
score = np.mean(scores)
knn.fit(nXtrain, Ytrain)
predYtest = knn.predict(nXtest)
poverty = pd.DataFrame(predYtest)
poverty.to_csv("submission.csv", header = ["Target"], index_label = 'Id')