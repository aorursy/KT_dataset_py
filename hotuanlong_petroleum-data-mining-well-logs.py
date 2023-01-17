import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn import metrics

import matplotlib.pyplot as plt

import random

import xgboost as xgb

from sklearn.svm import SVC
input_data = "../input/forcedataset/force-ai-well-logs/train.csv"

test_data = "../input/forcedataset/force-ai-well-logs/test.csv"
TARGET_1 = "FORCE_2020_LITHOFACIES_LITHOLOGY" # Class label

TARGET_2 = "FORCE_2020_LITHOFACIES_CONFIDENCE"

WELL_NAME = "WELL"
lithology_keys = {30000: 0,

                 65030: 1,

                 65000: 2,

                 80000: 3,

                 74000: 4,

                 70000: 5,

                 70032: 6,

                 88000: 7,

                 86000: 8,

                 99000: 9,

                 90000: 10,

                 93000: 11}
data_frame = pd.read_csv(input_data, sep=';')

test = pd.read_csv(test_data, sep=';')
data_frame.head()
# Numbers of rock types

rocks = data_frame[TARGET_1].value_counts()

rocks.shape
# Numbers of wells

wells = data_frame[WELL_NAME].value_counts()

wells.shape
data_frame.isna().sum()
unused_columns = [WELL_NAME, TARGET_1, TARGET_2, 'GROUP', 'FORMATION']

use_columns = [c for c in data_frame.columns if c not in unused_columns]
for c in use_columns:

    data_frame[c].fillna(data_frame[c].mean(), inplace=True)
data_frame.isna().sum()
data_frame[WELL_NAME].unique(), data_frame[WELL_NAME].unique().shape
random.shuffle(data_frame[WELL_NAME].unique()) 

train_wells = [] 

for i in range(78): 

    train_wells.append(data_frame[WELL_NAME].unique()[i]) 

train_mask = data_frame[WELL_NAME].isin(train_wells)
X_train = data_frame[train_mask][use_columns].values

y_train = data_frame[train_mask][TARGET_1].values

X_train.shape, y_train.shape
X_test = data_frame[~train_mask][use_columns].values

y_test = data_frame[~train_mask][TARGET_1].values
model = xgb.XGBClassifier(learning_rate = 0.1, max_depth=10, n_estimators=200, min_child_weight=10)

model
model2 = xgb.XGBClassifier(base_score = 0.7)

model2
model.fit(X_train, y_train)

model2.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred2 = model2.predict(X_test)
cm1 = confusion_matrix(y_true=y_test, y_pred=y_pred)



disp = metrics.plot_confusion_matrix(model, X_test, y_test)

disp.figure_.suptitle("Confusion Matrix")

plt.show()

print(cm1)
cm2 = confusion_matrix(y_true=y_test, y_pred=y_pred2)



disp = metrics.plot_confusion_matrix(model2, X_test, y_test)

disp.figure_.suptitle("Confusion Matrix")

plt.show()

print(cm2)
tp = 0

max = 0

for i in range(len(cm1)):

    tp += cm1[i][i]

    if cm1[i][i] > max:

        max = cm1[i][i]

print("Accuracy = {}".format(1.0*tp/np.sum(cm1)))
tp = 0

max = 0

for i in range(len(cm2)):

    tp += cm2[i][i]

    if cm2[i][i] > max:

        max = cm2[i][i]

print("Accuracy = {}".format(1.0*tp/np.sum(cm2)))
A = np.load('../input/penalty-matrix/penalty_matrix.npy')

A
def score(y_true, y_pred):

    S = 0.0

    y_true = y_true.astype(int)

    y_pred = y_pred.astype(int)

    for i in range(0, y_true.shape[0]):

        S -= A[lithology_keys[y_true[i]], lithology_keys[y_pred[i]]]

    return S/y_true.shape[0]
score(y_test, y_pred)
importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_train.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the impurity-based feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],

        color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), indices)

plt.xlim([-1, X_train.shape[1]])

plt.show()