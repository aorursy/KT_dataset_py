%matplotlib inline

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import Image

import pickle

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn import cross_validation
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/mushrooms.csv')
df.head(3)
class_map = {'p': 0, 'e': 1}

df['class'] = df['class'].map(class_map).astype(int)

# df['class'].as_matrix().flatten()

cap_shape_map = {'b': 1, 'c': 2, 'x': 3, 'f': 4, 'k': 5, 's': 6}

df['cap-shape'] = df['cap-shape'].map(cap_shape_map).astype(int)



cap_surface_map = {'f': 1, 'g': 2, 'y': 3, 's': 4}

df['cap-surface'] = df['cap-surface'].map(cap_surface_map).astype(int)



cap_color_map = {'w': 0, 'n': 1, 'b': 2, 'c': 3, 'g': 4, 'r': 5, 'p': 6, 'u': 7, 'e': 8, 'y': 9}

df['cap-color'] = df['cap-color'].map(cap_color_map).astype(int)



bruises_map = {'t': 1, 'f': 0}

df['bruises'] = df['bruises'].map(bruises_map).astype(int)



odor_map = {'n': 0, 'a': 1, 'l': 2, 'c': 3, 'y': 4, 'f': 5, 'm': 6, 'p': 7, 's': 8}

df['odor'] = df['odor'].map(odor_map).astype(int)



gill_attachment_map = {'a': 1, 'd': 2, 'f': 3, 'n': 4}

df['gill-attachment'] = df['gill-attachment'].map(gill_attachment_map).astype(int)



gill_spacing_map = {'c': 1, 'w': 2, 'd': 3}

df['gill-spacing'] = df['gill-spacing'].map(gill_spacing_map).astype(int)



gill_size_map = {'b': 1, 'n': 0}

df['gill-size'] = df['gill-size'].map(gill_size_map).astype(int)



gill_color_map = {'w': 0, 'k': 1, 'n': 2, 'b': 3, 'h': 4, 'g': 5, 'r': 6, 'o': 7, 'p': 8, 'u': 9, 'e': 10, 'y': 11}

df['gill-color'] = df['gill-color'].map(gill_color_map).astype(int)



stalk_shape_map = {'e': 1, 't': 0}

df['stalk-shape'] = df['stalk-shape'].map(stalk_shape_map).astype(int)



stalk_root_map = {'b': 1, 'c': 2, 'u': 3, 'e': 4, 'z': 5, 'r': 6, '?': 0}

df['stalk-root'] = df['stalk-root'].map(stalk_root_map).astype(int)



stalk_surface_above_ring_map = {'f': 1, 'y': 2, 'k': 3, 's': 4}

df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].map(stalk_surface_above_ring_map).astype(int)



stalk_surface_below_ring_map = {'f': 1, 'y': 2, 'k': 3, 's': 4}

df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].map(stalk_surface_above_ring_map).astype(int)



stalk_color_above_ring_map = {'w': 0, 'n': 1, 'b': 2, 'c': 3, 'g': 4, 'o': 5, 'p': 6, 'e': 7, 'y': 8}

df['stalk-color-above-ring'] = df['stalk-color-above-ring'].map(stalk_color_above_ring_map).astype(int)



stalk_color_below_ring_map = {'w': 0, 'n': 1, 'b': 2, 'c': 3, 'g': 4, 'o': 5, 'p': 6, 'e': 7, 'y': 8}

df['stalk-color-below-ring'] = df['stalk-color-below-ring'].map(stalk_color_below_ring_map).astype(int)



veil_type_map = {'p': 1, 'u': 0}

df['veil-type'] = df['veil-type'].map(veil_type_map).astype(int)



veil_color_map = {'w': 0, 'n': 1, 'o': 2, 'y': 3}

df['veil-color'] = df['veil-color'].map(veil_color_map).astype(int)



ring_number_map = {'n': 0, 'o': 1, 't': 2}

df['ring-number'] = df['ring-number'].map(ring_number_map).astype(int)



ring_type_map = {'n': 0, 'c': 1, 'e': 2, 'f': 3, 'l': 4, 'p': 5, 's': 6, 'z': 7}

df['ring-type'] = df['ring-type'].map(ring_type_map).astype(int)



spore_print_color_map = {'w': 0, 'k': 1, 'n': 2, 'b': 3, 'h': 4, 'r': 5, 'o': 6, 'u': 7, 'y': 8}

df['spore-print-color'] = df['spore-print-color'].map(spore_print_color_map).astype(int)



population_map = {'a': 1, 'c': 2, 'n': 3, 's': 4, 'v': 5, 'y': 6}

df['population'] = df['population'].map(population_map).astype(int)



habitat_map = {'g': 1, 'l': 2, 'm': 3, 'p': 4, 'u': 5, 'w': 6, 'd': 7}

df['habitat'] = df['habitat'].map(habitat_map).astype(int)

values = df.values

X = values[0::, 1::]

y = values[0::, 0]



print(X)

print(y)
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



acc_dict = {}
for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    print("======================")

    print(X_train, ">>>", y_train)

    print("---------------")

    print(X_test, "<<<<<", y_test)
for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]



    for clf in classifiers:

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)



candidate_classifier = SVC()

candidate_classifier.fit(X_train, Y_train)

result = candidate_classifier.predict(X_test)



print("Accuracy:", accuracy_score(Y_test, result))