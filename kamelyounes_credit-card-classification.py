# python libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

# classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

# pre-processing

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler





data =  pd.read_csv("../input/creditcardfraud/creditcard.csv")
data.head()
print('No Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]



sns.countplot('Class', data=data, palette=colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
print("ammount of null values :", data.isnull().sum().max())
valid_transactions = data.loc[data.Class == 0]

fraudulent_transactions = data.loc[data.Class == 1]



subset_valid_transactions = valid_transactions.sample(n=492)

balanced_data = (pd.concat([fraudulent_transactions, subset_valid_transactions])).sample(frac=1, axis=0).reset_index(drop=True)

print("fraudulent transactions count :", len(balanced_data.loc[balanced_data.Class == 1]))

print("valid transactions count :", len(balanced_data.loc[balanced_data.Class == 0]))

balanced_data['scaled_amount'] = StandardScaler().fit_transform(balanced_data['Amount'].values.reshape(-1,1))

balanced_data['scaled_time'] = StandardScaler().fit_transform(balanced_data['Time'].values.reshape(-1,1))



balanced_data.drop(['Time','Amount'], axis=1, inplace=True)
X = balanced_data.drop('Class', 1)

y = balanced_data['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

search_logistic_regression = GridSearchCV(LogisticRegression(), log_reg_params, cv=5)

search_logistic_regression.fit(X_train, y_train)

print("log reg best score :", search_logistic_regression.best_score_)





knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

search_k_neighbours = GridSearchCV(KNeighborsClassifier(), knears_params)

search_k_neighbours.fit(X_train, y_train)

print("kn best score :", search_k_neighbours.best_score_)
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

search_SVC = GridSearchCV(SVC(), svc_params)

search_SVC.fit(X_train, y_train)

print("svc  best score :", search_SVC.best_score_)

decision_tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}

search_decision_tree = GridSearchCV(DecisionTreeClassifier(), decision_tree_params)

search_decision_tree.fit(X_train, y_train)

print("decision tree best score :", search_decision_tree.best_score_)

log_reg = search_decision_tree.best_estimator_

print("log reg score :", log_reg.score(X_test, y_test))



svc = search_SVC.best_estimator_

print("svc score :", svc.score(X_test, y_test))
