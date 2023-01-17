# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/glass.csv")
data.head()
data.describe()
data["Type"].value_counts()
data.shape
sns.relplot(x="RI", y="Type", data=data)
sns.relplot(x="Na", y="Type", data=data)
sns.relplot(x="Mg", y="Type", data=data)
sns.relplot(x="Al", y="Type", data=data)
sns.relplot(x="Si", y="Type", data=data)
sns.relplot(x="K", y="Type", data=data)
sns.relplot(x="Ca", y="Type", data=data)
sns.relplot(x="Ba", y="Type", data=data)
sns.relplot(x="Fe", y="Type", data=data)
g = sns.PairGrid(data)
g.map(plt.scatter)
y = data["Type"]
X = data.drop(["Type"], axis=1)
numerical_features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
numerical_transformer = Pipeline(steps=[
    ("std_scaler", StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features)
])
def evaluate_classifier(label, classifier):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    pipeline.fit(X_train, y_train)
    print(label, ":", pipeline.score(X_test, y_test))
    y_pred = pipeline.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
evaluate_classifier("Decision Tree",
                    DecisionTreeClassifier(random_state=42))
evaluate_classifier("K Neighbors",
                    KNeighborsClassifier())
evaluate_classifier("Support Vector Machine",
                    SVC(random_state=42))
evaluate_classifier("Random Forest",
                    RandomForestClassifier(random_state=42))
evaluate_classifier("Multi-Layer Perceptron",
                    MLPClassifier(random_state=42))
evaluate_classifier("Random Forest, n=5",
                    RandomForestClassifier(random_state=42, n_estimators=5))
evaluate_classifier("Random Forest, n=10",
                    RandomForestClassifier(random_state=42, n_estimators=10))
evaluate_classifier("Random Forest, n=20",
                    RandomForestClassifier(random_state=42, n_estimators=20))
evaluate_classifier("Random Forest, n=50",
                    RandomForestClassifier(random_state=42, n_estimators=50))
evaluate_classifier("Random Forest, n=100",
                    RandomForestClassifier(random_state=42, n_estimators=100))
evaluate_classifier("Random Forest, n=200",
                    RandomForestClassifier(random_state=42, n_estimators=200))
evaluate_classifier("MLP Classifier, hidden_layer_size=20",
                   MLPClassifier(random_state=42, hidden_layer_sizes=(20,)))
evaluate_classifier("MLP Classifier, hidden_layer_size=50",
                   MLPClassifier(random_state=42, hidden_layer_sizes=(50,)))
evaluate_classifier("MLP Classifier, hidden_layer_size=100",
                   MLPClassifier(random_state=42, hidden_layer_sizes=(100,)))
evaluate_classifier("MLP Classifier, hidden_layer_size=200",
                   MLPClassifier(random_state=42, hidden_layer_sizes=(200,)))
evaluate_classifier("MLP Classifier",
                   MLPClassifier(random_state=42,
                                 hidden_layer_sizes=(80,80,80),
                                alpha=1))
