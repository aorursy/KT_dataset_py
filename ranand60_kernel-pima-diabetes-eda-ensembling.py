# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("darkgrid")
plt.style.use("classic")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
diabetes_path = '../input/pima-indians-diabetes-database/diabetes.csv'
diabetes_data = pd.read_csv(diabetes_path)

diabetes_data.head()
diabetes_data.info()
diabetes_data.describe()
plt.figure(figsize=(30,20))
sns.pairplot(diabetes_data, hue='Outcome')
plt.figure(figsize=(30,20))
sns.pairplot(diabetes_data, hue='Outcome', diag_kind='hist')
sns.catplot('Outcome', data=diabetes_data, kind='count')
feature_data = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_data[feature_data]
y = diabetes_data.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

def print_score(model, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = model.predict(X_train)
        print("Train Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, model.predict(X_train))}\n")
        
    elif train==False:
        pred = model.predict(X_test)
        print("Test Result:\n===========================================")        
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
tree = DecisionTreeClassifier()
bagging_model = BaggingClassifier(base_estimator=tree, n_estimators=1500, random_state=1)
bagging_model.fit(X_train, y_train)
print_score(bagging_model, X_train, y_train, X_test, y_test, train=True)
print_score(bagging_model, X_train, y_train, X_test, y_test, train=False)
rand_forest_model = RandomForestClassifier(random_state=1, n_estimators=1000)
rand_forest_model.fit(X_train, y_train)
print_score(rand_forest_model, X_train, y_train, X_test, y_test, train=True)
print_score(rand_forest_model, X_train, y_train, X_test, y_test, train=False)