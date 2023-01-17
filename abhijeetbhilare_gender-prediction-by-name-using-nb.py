import numpy as np 

import pandas as pd

import os

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("/kaggle/input/indonesian-names/indonesian-names.csv")

df.head()
df.columns
df.isnull().sum()
df.info()
print(df.gender.unique())

df = df.replace(' m', 'm')

df = df[df.gender != "LK"]

df = df[df.gender != "P"]

print(df.gender.unique())
df.gender.value_counts().plot(kind="bar")

df.gender.value_counts()
df.gender.replace({'f':0,'m':1},inplace=True)

df.head()
y = df.gender

x = df.name
cv = CountVectorizer()

X = cv.fit_transform(x)

cv.get_feature_names()[:5]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(x_train,y_train)

clf.score(x_test,y_test)
print("Validation Accuracy",clf.score(x_test,y_test)*100,"%")
print("Training Accuracy",clf.score(x_train,y_train)*100,"%")
plot_confusion_matrix(clf, x_test, y_test)
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes"]



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    MultinomialNB()]



for name, clf in zip(names, classifiers):

    print(name,   clf)

    clf.fit(x_train, y_train)

    score = clf.score(x_test, y_test)

    print("Validation Accuracy",score*100,"%")

import xgboost as xgb

dt = xgb.DMatrix(x_train,label=y_train)

dv = xgb.DMatrix(x_test,label=y_test)

params = {

    "eta": 0.2,

    "max_depth": 4,

    "objective": "binary:logistic",

    "silent": 1,

    "base_score": np.mean(y_train),

    'n_estimators': 1000,

    "eval_metric": "logloss"

}

model = xgb.train(params, dt, 2000, [(dt, "train"),(dv, "valid")], verbose_eval=200)

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, (y_pred>0.5))

print(cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

ax = sns.heatmap(cm, linewidth=0.5)

plt.show()

print("xgboost Acc : ", predict_accuracy_on_test_set)