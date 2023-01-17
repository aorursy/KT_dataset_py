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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns #for plotting

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split #for data splitting



np.random.seed(42)
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.info()
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 

              'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 

              'max_heart_rate_achieved', 'exercise_induced_angina', 

              'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 

              'target']
df.head()
df.dtypes
df.dtypes
df = pd.get_dummies(df, drop_first=True)
df.head(10)

X = df.drop('target', axis=1)

y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)



clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
def evaluate_preds(y_true, y_preds):

    """

    Performs comparison of evaluation metrics on y_true labels vs. 

    y_pred labels on a classification learning task

    """

    accuracy = accuracy_score(y_true, y_preds)

    precision = precision_score(y_true, y_preds)

    recall = recall_score(y_true, y_preds)

    f1 = f1_score(y_true, y_preds)

    metric_dict = {'accuracy': round(accuracy, 2),

                   'precision': round(precision, 2),

                   'recall': round(recall, 2),

                   'f1': round(f1, 2)}

    print(f"Acc: {accuracy * 100:.2f}%")

    print(f"Precision: {precision:.2f}")

    print(f"Recall: {recall:.2f}")

    print(f"f1: {f1:.2f}")

    

    return metric_dict

evaluate_preds(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

confusion_matrix
from sklearn.model_selection import RandomizedSearchCV



rf_param_grid = {

                 'max_depth' : [2, 4, 6, 8, 10],

                 'n_estimators': range(1,50),

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10, 20],

                 'min_samples_leaf': [1, 3, 10, 18],

                 'bootstrap': [True, False],

                 }
rf = RandomForestClassifier()
rf_rscv = RandomizedSearchCV(param_distributions=rf_param_grid,

                             estimator=rf, scoring='accuracy',

                             verbose=0, n_iter=100, cv=10)
rf_rscv.fit(X, y)
rf_rscv.best_score_
rf_rscv.best_params_
clf2 = RandomForestClassifier(n_estimators=36,

                             min_samples_split=3,

                             min_samples_leaf=10,

                             max_features='auto',

                             max_depth=4,

                             bootstrap=False)



clf2.fit(X_train, y_train)



y_pred2 = clf2.predict(X_test)
evaluate_preds(y_test, y_pred2)