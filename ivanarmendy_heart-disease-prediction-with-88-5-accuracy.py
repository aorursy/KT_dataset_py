# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install jcopml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head()
df.target.value_counts()
X = df.drop(columns="target")
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(scaling="standard"), ["age","trestbps","chol","thalach","oldpeak"]),
    ('categoric', cat_pipe(encoder='onehot'), ["sex","cp","fbs","restecg","exang","slope","ca","thal"]),
])
from sklearn.svm import SVC
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', SVC(max_iter=500))
])
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

model = RandomizedSearchCV(pipeline, rsp.svm_params, cv=3,  n_iter=100, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(scaling="robust",poly=3), ["age","trestbps","chol","thalach","oldpeak"]),
    ('categoric', cat_pipe(encoder='onehot'), ["sex","cp","fbs","restecg","exang","slope","ca","thal"]),
])
from sklearn.neighbors import KNeighborsClassifier
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', KNeighborsClassifier())
])
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

model = RandomizedSearchCV(pipeline, rsp.knn_poly_params, cv=3,  n_iter=100, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(scaling="standard"), ["age","trestbps","chol","thalach","oldpeak"]),
    ('categoric', cat_pipe(encoder='onehot'), ["sex","cp","fbs","restecg","exang","slope","ca","thal"]),
])
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

model = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=3,  n_iter=100, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(poly=3), ["age","trestbps","chol","thalach","oldpeak"]),
    ('categoric', cat_pipe(encoder='onehot'), ["sex","cp","fbs","restecg","exang","slope","ca","thal"]),
])
from xgboost import XGBClassifier
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', XGBClassifier(n_jobs=-1, random_state=42))
])
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

model = RandomizedSearchCV(pipeline, rsp.xgb_poly_params, cv=3,  n_iter=100, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(poly=2), ["age","trestbps","chol","thalach","oldpeak"]),
    ('categoric', cat_pipe(encoder='onehot'), ["sex","cp","fbs","restecg","exang","slope","ca","thal"]),
])
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))
])
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

model = RandomizedSearchCV(pipeline, rsp.rf_poly_params, cv=3,  n_iter=100, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
import matplotlib.pyplot as plt
import seaborn as sns
correlation = df.corr()
correlation['target'].sort_values(ascending=False)
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="cp", hue="target", data=df)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="sex", hue="target", data=df)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="fbs", hue="target", data=df)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="restecg", hue="target", data=df)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="exang", hue="target", data=df)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="slope", hue="target", data=df)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="ca", hue="target", data=df)
plt.show()
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="thal", hue="target", data=df)
plt.show()
df["highrisk"] = (df.exang == 1) &  (df.ca != 0) & (df.thal != 2) 
df.head()
X = df.drop(columns="target")
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(scaling="standard"), ["age","trestbps","chol","thalach","oldpeak"]),
    ('categoric', cat_pipe(encoder='onehot'), ["sex","cp","fbs","restecg","exang","slope","ca","thal","highrisk"]),
])
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

model = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=3,  n_iter=100, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
plt.figure(figsize=(7,6))
sns.distplot(df.trestbps[df.target == 0], color="r", label="heart disease")
sns.distplot(df.trestbps[df.target == 1], color="g", label="no heart disease")
plt.legend();
plt.figure(figsize=(7,6))
sns.distplot(df.age[df.target == 0], color="r", label="heart disease")
sns.distplot(df.age[df.target == 1], color="g", label="no heart disease")
plt.legend();
plt.figure(figsize=(7,6))
sns.distplot(df.chol[df.target == 0], color="r", label="heart disease")
sns.distplot(df.chol[df.target == 1], color="g", label="no heart disease")
plt.legend();
plt.figure(figsize=(7,6))
sns.distplot(df.thalach[df.target == 0], color="r", label="heart disease")
sns.distplot(df.thalach[df.target == 1], color="g", label="no heart disease")
plt.legend();
plt.figure(figsize=(7,6))
sns.distplot(df.oldpeak[df.target == 0], color="r", label="heart disease")
sns.distplot(df.oldpeak[df.target == 1], color="g", label="no heart disease")
plt.legend();
df["highrisk2"] = (df.trestbps > 140) &  (df.age > 50) & (df.chol > 250) & (df.thalach < 150) & (df.oldpeak > 0)
df.head()
X = df.drop(columns="target")
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(scaling="standard",poly=2), ["age","trestbps","chol","thalach","oldpeak"]),
    ('categoric', cat_pipe(encoder='onehot'), ["sex","cp","fbs","restecg","exang","slope","ca","thal","highrisk","highrisk2"]),
])
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', LogisticRegression(solver='sag', n_jobs=-1, random_state=42))
])
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

model = RandomizedSearchCV(pipeline, rsp.logreg_poly_params, cv=3,  n_iter=100, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
df_imp = mean_score_decrease(X_train, y_train, model, plot=True, topk=10)
from jcopml.plot import plot_classification_report, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
plot_roc_curve(X_train, y_train, X_test, y_test, model)
plot_pr_curve(X_train, y_train, X_test, y_test, model)