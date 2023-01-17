# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.info()
data.head()
plt.hist(data.target[data.target == 0], color = "green", label = "negative")

plt.hist(data.target[data.target == 1], color = "red", label = "positive")

plt.title("Diagnose distribution")

plt.legend()
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(data.loc[:,"age":"thal"], data.target, test_size = 0.2, random_state = 0)
traindata = pd.concat([xtrain, ytrain], axis = 1)
print("distribution by features and its target")

firstbox = plt.figure(figsize = (15, 15))

for i, col in enumerate(traindata):

    plt.subplot(4, 4, i + 1)

    sns.boxplot(x = traindata.target, y = traindata[col])

firstbox.tight_layout(h_pad = 2)

target0sex = [traindata[(traindata.target == 0) & (traindata.sex == 0)].shape[0], traindata[(traindata.target == 0) & (traindata.sex == 1)].shape[0]]

target1sex = [traindata[(traindata.target == 1) & (traindata.sex == 0)].shape[0], traindata[(traindata.target == 1) & (traindata.sex == 1)].shape[0]]
sextarget = pd.DataFrame({"negative": target0sex, "positive": target1sex})

sextarget.index = ["sex value 0", "sex value 1"]

sextarget
correlation = traindata.corr()
targetcorr = correlation.target.sort_values()

targetcorr
plt.figure(figsize = (17, 7))

sns.distplot(traindata.loc[traindata.target == 0, "cp"], label = "negative", color = "green", kde_kws={"bw": 0.3}, hist = False)

sns.distplot(traindata.loc[traindata.target == 1, "cp"], label = "positive", color = "red", kde_kws = {"bw":0.3}, hist = False)

plt.title("Distribution on cp")
plt.figure(figsize = (17, 7))

sns.distplot(traindata.loc[traindata.target == 0, "thalach"], label = "negative", color = "green", kde_kws={"bw": 6}, hist = False)

sns.distplot(traindata.loc[traindata.target == 1, "thalach"], label = "positive", color = "red", kde_kws = {"bw":6}, hist = False)

plt.title("Distribution on thalach")
smallimpact = targetcorr[abs(targetcorr) < 0.2]

featureeng = list(smallimpact.index)

featureeng
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 3)

polycolm = traindata.loc[:, featureeng]

poly.fit(polycolm)

trainpoly = poly.transform(polycolm)

trainpoly.shape
poly_features = pd.DataFrame(trainpoly, columns=poly.get_feature_names(featureeng))

poly_features.index = traindata.index

poly_features["target"] = traindata.target

poly_features
poly_corr = poly_features.corr()["target"].sort_values()

poly_corr
from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree, gaussian_process

MLA = [

       ensemble.AdaBoostClassifier(),

       ensemble.BaggingClassifier(),

       ensemble.GradientBoostingClassifier(),

       ensemble.RandomForestClassifier(),

       gaussian_process.GaussianProcessClassifier(),

       linear_model.LogisticRegressionCV(),  

       linear_model.SGDClassifier(),

       naive_bayes.BernoulliNB(),

       naive_bayes.GaussianNB(),

       neighbors.KNeighborsClassifier(),

       svm.SVC(probability=True),

       svm.NuSVC(probability=True),

       svm.LinearSVC(),

       tree.DecisionTreeClassifier(),

       tree.ExtraTreeClassifier(),

]
name = []

testscore = []

for alg in MLA:

    name.append(alg.__class__.__name__)

    alg.fit(xtrain, ytrain)

    testscore.append(alg.score(xtest, ytest))

    

vergleich = pd.DataFrame({"name": name, "testscore": testscore})
vergleich = vergleich.sort_values(by = "testscore", ascending = False)

vergleich