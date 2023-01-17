import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import auc, roc_auc_score, f1_score, confusion_matrix

!pip install -q scikit-plot

import scikitplot as skplot

from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

import warnings

import os

warnings.filterwarnings(action='ignore')
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()
print(f"Training data shape : {data.shape}")
data.Class.value_counts()
plt.figure(figsize=(5,5))

ax = sns.countplot(x= data.Class, data=data, hue="Class")

for p in ax.patches:

  percentage = '{:.1f}%'.format(100 * p.get_height()/data.shape[0])

  x = p.get_x() + p.get_width() / 2 - 0.15

  y = p.get_y() + p.get_height()

  ax.annotate(percentage, (x, y), size = 12)

plt.show();
data.isnull().sum()
data.Time.describe()
std_sclar = StandardScaler()

std_sclar.fit(data.Time.values.reshape(-1,1))

data["std_time"] = std_sclar.transform(data.Time.values.reshape(-1,1))

std_sclar.fit(data.Amount.values.reshape(-1,1))

data["std_amount"] = std_sclar.transform(data.Amount.values.reshape(-1,1))
data.drop(labels=['Time','Amount'], axis=1, inplace=True)
data.head()
y = data.Class

X = data.drop(labels=['Class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train Data 0's {round(len(y_train[y_train == 0])/len(y_train) * 100, 2)} % and 1's {round(len(y_train[y_train == 1])/len(y_train) * 100, 2)} %")

print(f"Test Data 0's {round(len(y_test[y_test == 0])/len(y_test) * 100, 2)} % and 1's {round(len(y_test[y_test == 1])/len(y_test) * 100, 2)} %")
lrg = LogisticRegression(class_weight='balanced')

gridcv = GridSearchCV(estimator=lrg, param_grid={'C':[0.01,0.1,1,10,100]}, n_jobs=-1, scoring='roc_auc', verbose=1, cv=3)

gridcv.fit(X_train, y_train)

print(gridcv.best_score_)

print(gridcv.best_estimator_)
model = LogisticRegression(C=0.01, class_weight='balanced', dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)

model.fit(X_train, y_train)

y_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

print(f"Train AUC Score : {roc_auc_score(y_train, y_pred)}")

print(f"Test AUC Score : {roc_auc_score(y_test, y_test_pred)}")
skplot.metrics.plot_confusion_matrix(y_test, y_test_pred, figsize=(7,5))
over_samples = SMOTE(random_state=42, n_jobs=-1)

X_train_over, y_train_over = over_samples.fit_sample(X_train, y_train)
print(f"After over sampling Positive Samples Count : {len(y_train_over[y_train_over == 1])}")

print(f"After over sampling Negitive Samples Count : {len(y_train_over[y_train_over == 0])}")
model = LogisticRegression(C=0.01, dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)

model.fit(X_train_over, y_train_over)

y_pred = model.predict(X_train_over)

y_test_pred = model.predict(X_test)

print(f"Train AUC Score : {roc_auc_score(y_train_over, y_pred)}")

print(f"Test AUC Score : {roc_auc_score(y_test, y_test_pred)}")
skplot.metrics.plot_confusion_matrix(y_test, y_test_pred)
rus = RandomUnderSampler(random_state=42)

X_train_under, y_train_under = rus.fit_sample(X_train, y_train)
print(f"After under sampling Positive Samples Count : {len(y_train_under[y_train_under == 1])}")

print(f"After under sampling Negitive Samples Count : {len(y_train_under[y_train_under == 0])}")
model = LogisticRegression(C=0.01, dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)

model.fit(X_train_under, y_train_under)

y_pred = model.predict(X_train_under)

y_test_pred = model.predict(X_test)

print(f"Train AUC Score : {roc_auc_score(y_train_under, y_pred)}")

print(f"Test AUC Score : {roc_auc_score(y_test, y_test_pred)}")
skplot.metrics.plot_confusion_matrix(y_test, y_test_pred)