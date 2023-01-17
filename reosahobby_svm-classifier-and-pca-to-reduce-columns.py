# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.options.display.max_columns = 100
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.drop(columns="Unnamed: 32", inplace=True)
df.head()
plot_missing_value(df, return_df=True)
df.shape
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x=df["diagnosis"])
plt.figure(figsize=(20, 10))
ax = sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f')
df.describe()
X = df.drop(columns=["diagnosis", "id"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.decomposition import PCA
pca = PCA().fit(X_train)

plt.figure(figsize=(14,5))
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.xlabel("n_components")
plt.ylabel("Comulative explained variance")
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp
from jcopml.tuning.space import Integer, Real
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(impute='median', poly=2), X_train.columns)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('pca', PCA(whiten=True)),
    ('algo', SVC(max_iter=500))
])
parameter = {
    'prep__numeric__poly__degree': [2],
    'pca__n_components' : [10],
    'algo__gamma': Real(low=-3, high=3, prior='log-uniform'),
    'algo__C': Real(low=-3, high=3, prior='log-uniform')
}


model = RandomizedSearchCV(pipeline, parameter, cv=3,  n_iter=100, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
print("\nTrain Akurasi:", model.score(X_train, y_train))
print("Test Akurasi:", model.score(X_test, y_test))
from jcopml.plot import plot_classification_report, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
plot_confusion_matrix(X_train, y_train, X_test, y_test, model)
plot_classification_report(X_train, y_train, X_test, y_test, model, report=True)