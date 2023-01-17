import pandas as pd

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
dataset = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

dataset = shuffle(dataset)

dataset.reset_index(inplace=True, drop=True)
dataset.head()
len(dataset)
cor = dataset.corr()

sns.heatmap(cor, vmax=1, vmin=-1)
abs(cor['DEATH_EVENT']).sort_values()[::-1]
x = dataset[['time','serum_creatinine','ejection_fraction','age','serum_sodium']]

y = dataset['DEATH_EVENT']



xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 1, test_size = 0.3 )
lr = LogisticRegression()

lr.fit(xtrain, ytrain)

lr_score=accuracy_score(ytest, lr.predict(xtest))

print("Logistic Regression Accuracy : {:.2f}%".format(lr_score))
gnb = GaussianNB()

gnb.fit(xtrain, ytrain)

gnb_score = accuracy_score(ytest, gnb.predict(xtest))

print("Navie Bayes Accuracy : {:.2f}%".format(gnb_score))
sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

sgd.fit(xtrain, ytrain)

sgd_score = accuracy_score(ytest, sgd.predict(xtest))

print("SGD Classifier Accuracy : {:.2f}%".format(gnb_score))
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(xtrain, ytrain)

knn_score = accuracy_score(ytest, knn.predict(xtest))

print("KNN Classifier Accuracy : {:.2f}%".format(knn_score))
dtc = DecisionTreeClassifier()

dtc.fit(xtrain, ytrain)

dtc_score = accuracy_score(ytest, dtc.predict(xtest))

print("Decision Tree Classifier Accuracy : {:.2f}%".format(dtc_score))
print("Logistic Regression Accuracy : {:.2f}%".format(lr_score))

print("Navie Bayes Accuracy : {:.2f}%".format(gnb_score))

print("SGD Classifier Accuracy : {:.2f}%".format(gnb_score))

print("KNN Classifier Accuracy : {:.2f}%".format(knn_score))

print("Decision Tree Classifier Accuracy : {:.2f}%".format(dtc_score))