# Some important libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
pulsar = pd.read_csv('../input/pulsar_stars.csv')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(pulsar.drop('target_class', axis=1))

scaled_feat = scaler.transform(pulsar.drop('target_class', axis=1))
from sklearn.model_selection import train_test_split, GridSearchCV

# GridSearchCV: cross validation (used to train the model)
X = pd.DataFrame(scaled_feat, columns=pulsar.columns[:-1])

y = pulsar['target_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
param = {'n_neighbors': [5, 10, 15, 20, 25, 30], 'p': [2, 3, 4, 5, 6]}

gsc = GridSearchCV(knn, param, cv=5, refit=True)

# cv=5: 5-fold Cross-validation

gsc.fit(X_train, y_train)
gsc.best_estimator_
gsc.best_params_
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, grid_predict))
plt.figure(figsize=(6,6))

cm = confusion_matrix(y_test,grid_predict)

sns.set(font_scale=1.25)

sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='Blues')

plt.title('Confusion matrix')