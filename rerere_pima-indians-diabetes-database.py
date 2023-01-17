# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Dataset : https://www.kaggle.com/uciml/pima-indians-diabetes-database

df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

df.head()
df.info()
df.describe().T
import eli5

from eli5.sklearn import PermutationImportance

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



# We get all the column except the last one.

X = df.iloc[:, :-1]

# We get the last column (Outcome)

y = df.iloc[:, -1]



# Create two sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



my_model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)



perm = PermutationImportance(my_model, random_state=1).fit(X, y)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
import matplotlib.pyplot as plt

import seaborn as sns



g = sns.kdeplot(df["Glucose"][(df["Outcome"] == 0) & (df["Glucose"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df["Glucose"][(df["Outcome"] == 1) & (df["Glucose"].notnull())], color="Blue", ax=g, shade = True)

g.set_xlabel("Glucose")

g.set_ylabel("Frequency")

g = g.legend(["No Diabete","Diabete"])
g = sns.pairplot(df, hue="Outcome", palette="Set2", diag_kind="kde", height=3, size=3)
from sklearn.base import BaseEstimator, TransformerMixin



# A class to select numerical or categorical columns 

# since Scikit-Learn doesn't handle DataFrames yet

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



from sklearn.preprocessing import MinMaxScaler



# No need of the imputer because all the value are there :)

preprocess_pipeline = Pipeline([

    ("select_numeric", DataFrameSelector(["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Pregnancies"])),

#   ("imputer", SimpleImputer(strategy="median")),

    ("Standardization", StandardScaler())

])



# Create two sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



X_train = preprocess_pipeline.fit_transform(X_train)

X_test = preprocess_pipeline.fit_transform(X_test)
from sklearn.svm import SVC



svm_clf = SVC(gamma="auto")

svm_clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score



# Here, we use cross validation

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)

svm_scores.mean()
from sklearn.model_selection import GridSearchCV



parameters = { 

    'gamma': [0.001, 0.01, 0.1, 1, 10], 

    'kernel': ['rbf'], 

    'C': [0.001, 0.01, 0.1, 1, 10, 15, 20],

}



svm_clf = GridSearchCV(SVC(), parameters, cv=10, n_jobs=-1).fit(X_train, y_train)
svm_clf.cv_results_['params'][svm_clf.best_index_]
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



y_pred = svm_clf.predict(X_test)

accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
from sklearn.model_selection import cross_val_score



svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)

svm_scores.mean()
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores.mean()
parameters = { 

    'n_estimators': [10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],

}



forest_clf = GridSearchCV(RandomForestClassifier(random_state=42), parameters, cv=10, n_jobs=-1).fit(X_train, y_train)



print(forest_clf.cv_results_['params'][forest_clf.best_index_])
forest_clf.fit(X_train, y_train)



y_pred = forest_clf.predict(X_test)



cm = confusion_matrix(y_test, y_pred)



print('Confusion matrix\n',cm)
from sklearn.metrics import plot_confusion_matrix





class_names = ["No Diabete", "Diabet"]



np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

titles_options = [("Confusion matrix, without normalization", None),

                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:

    disp = plot_confusion_matrix(forest_clf, X_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 normalize=normalize)

    disp.ax_.set_title(title)



    print(title)

    print(disp.confusion_matrix)



plt.show()

from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()



y_pred = gnb.fit(X_train, y_train).predict(X_test)

cm = confusion_matrix(y_test, y_pred)



print(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))



print('Confusion matrix\n', cm)





gnb_scores = cross_val_score(gnb, X_train, y_train, cv=10)

print(gnb_scores.mean())
from sklearn.neighbors import KNeighborsClassifier



neigh_clf = KNeighborsClassifier(n_neighbors=5)

y_pred = neigh_clf.fit(X_train, y_train).predict(X_test)



print(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))

print('Confusion matrix\n', cm)
knn2 = KNeighborsClassifier()

param_grid = {

    'n_neighbors': np.arange(1, 50)

}

knn_gscv = GridSearchCV(knn2, param_grid, cv=5).fit(X_train, y_train)
print(knn_gscv.best_params_)

print(knn_gscv.best_score_)
neigh_clf = KNeighborsClassifier(n_neighbors=21)

y_pred = neigh_clf.fit(X_train, y_train).predict(X_test)



print(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))

print('Confusion matrix\n', cm)





knn_scores = cross_val_score(neigh_clf, X_train, y_train, cv=10)

print(knn_scores.mean())
from sklearn.metrics import plot_roc_curve

ax = plt.gca()

forest_clf_roc_curve = plot_roc_curve(forest_clf, X_test, y_test, ax=ax, alpha=0.8)

svm_clf_roc_curve = plot_roc_curve(svm_clf, X_test, y_test, ax=ax, alpha=0.8)

bay_clf_curve = plot_roc_curve(gnb, X_test, y_test, ax=ax, alpha=0.8)

knn_clf_curve = plot_roc_curve(knn_gscv, X_test, y_test, ax=ax, alpha=0.8)

plt.show()