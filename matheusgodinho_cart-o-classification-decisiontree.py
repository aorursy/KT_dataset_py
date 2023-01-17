import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv', header=0)
import seaborn as sns



corr_matrix = df.corr()



fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corr_matrix, vmax=1.0, center=0, fmt='.2f',

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})

plt.show();
#Correlation with output variable

cor_target = abs(corr_matrix['default payment next month'])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.5]

relevant_features
features = df.drop('default payment next month', axis=1)

classification = df['default payment next month']

features.drop(columns=['ID'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, classification.values, test_size=0.25, random_state=42)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score
max_depths = np.linspace(1, 30, 30, endpoint=True)

train_results = []

test_results = []

for max_depth in max_depths:

   dt = DecisionTreeClassifier(max_depth=max_depth)

   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)

   roc_auc = roc_auc_score(y_train, train_pred)

   # Add auc score to previous train results

   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   roc_auc = roc_auc_score(y_test, y_pred)

   # Add auc score to previous test results

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Tree depth')

plt.show()
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []

test_results = []

for min_samples_split in min_samples_splits:

   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)

   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)

   roc_auc = roc_auc_score(y_train, train_pred)

   # Add auc score to previous train results

   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   roc_auc = roc_auc_score(y_test, y_pred)

   # Add auc score to previous test results

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')

line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Samples split')

plt.show()
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

train_results = []

test_results = []

for min_samples_leaf in min_samples_leafs:

   dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)

   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)

   roc_auc = roc_auc_score(y_train, train_pred)

   # Add auc score to previous train results

   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   roc_auc = roc_auc_score(y_test, y_pred)

   # Add auc score to previous test results

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')

line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Sample Leafs')

plt.show()
max_features = list(range(1,features.shape[1]))

train_results = []

test_results = []

for max_feature in max_features:

   dt = DecisionTreeClassifier(max_features=max_feature)

   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)

   roc_auc = roc_auc_score(y_train, train_pred)

   # Add auc score to previous train results

   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   roc_auc = roc_auc_score(y_test, y_pred)

   # Add auc score to previous test results

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Max Features')

plt.show()
tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100, min_samples_leaf=0.2, min_samples_split=0.1, max_depth=3)

tree.fit(X_train, y_train)

pred = tree.predict(X_test)

roc_auc_score(y_test, pred)
valid = pd.read_csv('../input/valid.csv', header=0)

test = pd.read_csv('../input/test.csv', header=0)





result = pd.DataFrame(columns=['ID', 'Default'])

result.ID = np.append(valid['ID'].values, test['ID'].values)



valid.drop(columns=['ID'], axis=1, inplace=True)

test.drop(columns=['ID'], axis=1, inplace=True)



predictedValid = tree.predict(valid)

predictedTest = tree.predict(test)

predictions = np.append(predictedValid, predictedTest)

result.Default = predictions

result.to_csv("output.csv", encoding="utf8", index=False)