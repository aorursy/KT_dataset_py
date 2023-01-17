# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


%load_ext autoreload
%autoreload 2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.float_format = '{:,.5f}'.format

from pandas_profiling import ProfileReport # library for automatic EDA

from time import time
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import datasets

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the data and displaying some rows
df = pd.read_csv("/kaggle/input/iris/Iris.csv").drop('Id', axis=1)

X = df.drop(['Species'], axis=1)
y = pd.Series(LabelEncoder().fit_transform(df['Species']))
X
report = ProfileReport(df)
display(report)
g = sns.pairplot(df, hue='Species', markers='x')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    random_state=1, test_size=0.33,
    shuffle=True, stratify=y
)

# Create prediction pipeline
# - Logistic Regression will converge faster if the features are scaled to the same range
scaler = StandardScaler() 
clf = LogisticRegression(random_state=1)

pipeline = make_pipeline(scaler, clf)

# Fit the model, predict train and test sets
pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Calculate scores, display results
display(
    f'Train accuracy: {accuracy_score(y_train, y_pred_train)}',
    f'Test accuracy: {accuracy_score(y_test, y_pred_test)}', 
)

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, train_sizes=np.arange(10, 110, 10),
    random_state=1, shuffle=True, scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(x=train_sizes, y=train_scores_mean, label='train accuracy', ax=ax)
sns.lineplot(x=train_sizes, y=test_scores_mean, label='test accuracy', ax=ax)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

res = cross_validate(pipeline, X, y, scoring='accuracy', return_train_score=True, cv=cv, n_jobs=-1)

train_score = np.mean(res['train_score'])
test_score = np.mean(res['test_score'])

display(
    f'Mean train accuracy: {train_score:.2f}',
    f'Mean test accuracy: {test_score:.2f}', 
)