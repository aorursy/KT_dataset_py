# Scikit-learn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import datasets

# Other libraries

import numpy as np

import pandas as pd

import scipy.sparse

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
# Use vector drawing inside jupyter notebook

%config InlineBackend.figure_format = "svg"

# Set matplotlib default axis font size (inside this notebook)

plt.rcParams.update({'font.size': 8})
iris = datasets.load_iris()

X = iris.data;

y = LabelEncoder().fit_transform(iris.target);
X = StandardScaler(with_mean=True,with_std=True).fit_transform(X);

[X_train,X_test,y_train,y_test] = train_test_split(X,y,stratify=y,random_state=123)
gnb = GaussianNB().fit(X_train,y_train);

print("Gaussian Naive Bayes Test accuracy score: {:.3f}".format(

        accuracy_score(y_test,gnb.predict(X_test))))
y = np.load("../input/test-data/20newsgroups_vectorized_target.npz",allow_pickle=True)
y_train,y_test,y_names = y["y_train"],y["y_test"],y["y_names"]
X_test = scipy.sparse.load_npz("../input/test-data/20newsgroups_vectorized_test_data.npz")

X_train = scipy.sparse.load_npz("../input/test-data/20newsgroups_vectorized_train_data.npz")
mnb = GridSearchCV(MultinomialNB(),{"alpha":np.arange(0.01,0.1,step=0.01)},

                   cv=5,error_score="f1_macro").fit(X_train,y_train);
print("Test Classification report\n",

      classification_report(y_test,mnb.predict(X_test),

                           target_names=y_names))