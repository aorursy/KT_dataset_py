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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split 

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFE

import time

from pprint import pprint

from tabulate import tabulate

from sklearn.tree import export_graphviz

import eli5

from eli5.sklearn import PermutationImportance
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df['age'].value_counts()
df = df[df['thal'] != 0]

df = df[df['age'] != 29]

df.head()
df['thal'] = df['thal'].replace(1, 'fixed defect')

df['thal'] = df['thal'].replace(2, 'normal')

df['thal'] = df['thal'].replace(3, 'reversable defect')

df['cp'] = df['cp'].replace(0, 'asymptomatic')

df['cp'] = df['cp'].replace(1, 'atypical angina')

df['cp'] = df['cp'].replace(2, 'non-anginal pain')

df['cp'] = df['cp'].replace(3, 'typical angina')

df['restecg'] = df['restecg'].replace(0, 'ventricular hypertrophy')

df['restecg'] = df['restecg'].replace(1, 'normal')

df['restecg'] = df['restecg'].replace(2, 'ST-T wave abnormality')

df['slope'] = df['slope'].replace(0, 'downsloping')

df['slope'] = df['slope'].replace(1, 'flat')

df['slope'] = df['slope'].replace(2, 'upsloping')
temp = pd.get_dummies(df[['cp', 'restecg', 'slope', 'thal']])

df = df.join(temp, how='left')

df = df.drop(columns = ['cp','restecg', 'slope', 'thal'], axis=1)

df.head()
df = df.drop(columns = ['restecg_ventricular hypertrophy', 'slope_upsloping', 'thal_fixed defect', 'cp_typical angina'], axis=1)

df.head()
df = df.drop_duplicates()

df.shape
df.corr()
ax = sns.heatmap(df, annot=True)
sns.factorplot('age', kind='count', hue='target', data=df, palette='coolwarm', height=10, aspect=.8)
plt.figure(figsize=(8,8))

sns.scatterplot(x=df['age'],y=df['thalach'],hue=df['target'])

plt.xlabel('age')

plt.ylabel('thalach')

plt.show()
sns.countplot(df.target)

df.target.value_counts()
features = ['age', 'ca', 'cp_asymptomatic', 'exang', 'oldpeak',

  'slope_flat', 'thal_normal', 'thal_reversable defect',

  'thalach', 'cp_non-anginal pain', 'trestbps',

  'sex', 'chol', 'restecg_normal', 'cp_atypical angina',

  'slope_downsloping', 'fbs','restecg_ST-T wave abnormality', 'target']



df = df[features]
X = df.drop(['target'], axis=1)

Y = df['target']

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=0)
rf = RandomForestClassifier()

dt = DecisionTreeClassifier()

lr = LogisticRegression()

knn = KNeighborsClassifier()
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [2],

    'min_samples_leaf': [5],

    'n_estimators': [200],

    'oob_score': [True],

    'random_state': [0],

}



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                      cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)



# Fit the grid search to the data

grid_search.fit(train_features, train_labels);



grid_search.best_params_

best_grid = grid_search.best_estimator_

pprint(best_grid.get_params())



selector = RFE(rf, step=1, verbose=3)

selector = selector.fit(train_features, train_labels)

print("Features sorted by their rank:")

pprint(sorted(zip(map(lambda x: round(x, 4), selector.ranking_), X)))
rf = RandomForestClassifier(**best_grid.get_params())

dt = DecisionTreeClassifier()

lr = LogisticRegression()

knn = KNeighborsClassifier()
rf.fit(train_features, train_labels)

dt.fit(train_features, train_labels)

lr.fit(train_features, train_labels)

knn.fit(train_features, train_labels)
rf_pred_train = rf.predict(train_features)

dt_pred_train = dt.predict(train_features)

lr_pred_train = lr.predict(train_features)

knn_pred_train = knn.predict(train_features)

rf_pred_test = rf.predict(test_features)

dt_pred_test = dt.predict(test_features)

lr_pred_test = lr.predict(test_features)

knn_pred_test = knn.predict(test_features)
rf_prob = rf.predict_proba(test_features)[:,1]

dt_prob = dt.predict_proba(test_features)[:,1]

lr_prob = lr.predict_proba(test_features)[:,1]

knn_prob = knn.predict_proba(test_features)[:,1]
rf_prob
print(classification_report(test_labels,rf_pred_test))

print('Random Forest baseline: ' + str(roc_auc_score(train_labels, rf_pred_train)))

print('Random Forest: ' + str(roc_auc_score(test_labels, rf_pred_test)))

print(classification_report(test_labels,dt_pred_test))

print('Decision Tree baseline: ' + str(roc_auc_score(train_labels, dt_pred_train)))

print('Decision Tree: ' + str(roc_auc_score(test_labels, dt_pred_test)))

print(classification_report(test_labels,lr_pred_test))

print('Logistic Regression baseline: ' + str(roc_auc_score(train_labels, lr_pred_train)))

print('Logistic Regression: ' + str(roc_auc_score(test_labels, lr_pred_test)))

print(classification_report(test_labels,knn_pred_test))

print('KNN baseline: ' + str(roc_auc_score(train_labels, knn_pred_train)))

print('KNN: ' + str(roc_auc_score(test_labels, knn_pred_test)))
ns_probs = [0 for _ in range(len(test_labels))]

ns_fpr, ns_tpr, _ = roc_curve(test_labels, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(test_labels, rf_prob)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

plt.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()
prec, rec, tre = precision_recall_curve(test_labels, rf_prob)

def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):

    fig, ax = plt.subplots(figsize=(10,6))

    plt.plot(thresholds, precisions[:-1], 'r', label='Precisions')

    plt.plot(thresholds, recalls[:-1], '#424242', label='Recalls')

    plt.ylabel('Level of Precision and Recall', fontsize=12)

    plt.title('Precision and Recall Scores as a function of the decision threshold', fontsize=12)

    plt.xlabel('Thresholds', fontsize=12)

    plt.legend(loc='best', fontsize=12)

    plt.ylim([0,1])

    plt.xlim([0,1])

    plt.axvline(x=0.5, linewidth=3, color='#0B3861')



plot_prec_recall_vs_tresh(prec, rec, tre)

plt.show()
perm = PermutationImportance(rf, random_state=1).fit(train_features, train_labels)

eli5.show_weights(perm, feature_names = X.columns.tolist())