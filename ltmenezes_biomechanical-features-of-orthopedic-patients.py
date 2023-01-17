import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import display

from time import time

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/column_2C_weka.csv')



data.info()

data.head()
import missingno

missingno.matrix(data)
f, ax = plt.subplots(figsize=(16, 10))

corr = data.corr()

corr_mtx = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=False, ax=ax, annot = True)
from numpy import mean



n_row = 3

n_col = 2

feature_variables = list(data.columns)[0:-1]

target_variable = list(data.columns)[-1]

f, axes = plt.subplots(n_row, n_col, figsize=(16, 12))

k = 0



for i in list(range(n_row)):

    for j in list(range(n_col)):

        sns.barplot(x = feature_variables[k], y = target_variable, data = data, estimator = mean, color = 'lightblue', ax=axes[i, j])

        k = k + 1
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures



scaler = MinMaxScaler()

features_raw = data.drop(columns=['class'])

target_raw = data['class']



types_aux = pd.DataFrame(features_raw.dtypes)

types_aux.reset_index(level=0, inplace=True)

types_aux.columns = ['Variable','Type']

numerical = list(types_aux[types_aux['Type'] == 'float64']['Variable'].values)



features_minmax_transform = pd.DataFrame(data = features_raw)

features_minmax_transform[numerical] = scaler.fit_transform(features_raw[numerical])



features_minmax_transform.head()
from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()



lb.fit(target_raw)

target = lb.transform(target_raw)

from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

import os



X = features_minmax_transform

y = target



clf_a = MultinomialNB()

clf_b = DecisionTreeClassifier(random_state = 0)

clf_c = RandomForestClassifier(random_state = 0)

clf_d = LogisticRegression(random_state = 0)

clf_e = SGDClassifier(random_state = 0)

clf_f = KNeighborsClassifier()



list_clf = [clf_a, clf_b, clf_c, clf_d, clf_e, clf_f]



results = []

for clf in list_clf:

    start = time()

    clf_name = clf.__class__.__name__

    cv = 5

    scoring = 'f1'



    scores_f1 = cross_val_score(clf, X, y, cv=cv, scoring = scoring)

    scores_ = cross_val_score(clf, X, y, cv=cv)

    end = time()

    train_time = end  - start

    results.append([clf_name, np.mean(scores_f1), np.mean(scores_), train_time])





df_results = pd.DataFrame(np.array(results))

df_results.columns = ['Classifier', 'F1-Score', 'Accuracy', 'Train Time']

df_results.sort_values(by=['F1-Score'], ascending=False)
results = []

for i in range(10, 300):

    rfc = RandomForestClassifier(random_state = 0, n_estimators=i)

    start = time()

    cv = 5

    scoring = 'f1'



    scores_f1 = cross_val_score(rfc, X, y, cv=cv, scoring = scoring)

    scores_ = cross_val_score(rfc, X, y, cv=cv)

    end = time()

    train_time = end  - start

    results.append(['RandomForestClassifier', np.mean(scores_f1), np.mean(scores_), train_time, i])

df_results = pd.DataFrame(np.array(results))

df_results.columns = ['Classifier', 'F1-Score', 'Accuracy', 'Train Time', 'n_estimators']

fig, ax1 = plt.subplots(figsize=(16, 10))



color = 'tab:red'

ax1.set_xlabel('n_estimators')

ax1.set_ylabel('Accuracy', color=color)

ax1.plot(range(10,300), df_results['Accuracy'], color=color)

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylabel('F1-Score', color=color)  # we already handled the x-label with ax1

ax2.plot(range(10,300), df_results['F1-Score'], color=color)

ax2.tick_params(axis='y', labelcolor=color)



fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
df_results["Accuracy"] = df_results.Accuracy.astype(float)

df_results.loc[df_results['Accuracy'].idxmax()]