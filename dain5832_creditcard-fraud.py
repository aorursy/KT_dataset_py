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
# import necessary modules
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, plot_roc_curve, roc_auc_score, precision_recall_curve, precision_score, recall_score, make_scorer

import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df
# get column name
print(df.columns)

# get data summary
print(df.describe()) 

# We can find that there's no na
# Also variables are all numeric, not categorical
# split dataset into x and y
X_all = df.drop("Class", axis=1)
y_all = df["Class"]
# get number of data for each class
y_all.value_counts()
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
# Since there's class imbalance, let's oversample data
X = pd.concat([X_train, y_train], axis=1)

not_fraud = X[X.Class==0]
fraud = X[X.Class==1]

# Oversampling
fraud_oversampled = resample(fraud, n_samples=len(not_fraud), replace=True, random_state=42)
X_oversampled = pd.concat([fraud_oversampled, not_fraud])

# Check
X_oversampled.Class.value_counts()
# devide into x and y
X_train = X_oversampled.drop("Class", axis=1)
y_train = X_oversampled["Class"]
# remove outliers (need to be implemented)
# demensionality reduction and clustering (need to be implemented)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
LR = lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
# see evaluation results
mse = mean_squared_error(y_test, lr_preds)
print(mse)

roc = roc_auc_score(y_test, lr_preds)
print(roc)

print(classification_report(y_test, lr_preds))

# draw plot
fig = plt.figure(figsize(10, 7))
axes1 = fig.add_subplot(1, 2, 1)
axes2 = fig.add_subplot(1, 2, 2)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, verbose=2, random_state=42)
RF = model.fit(X_train, y_train)
preds = model.predict(X_test)

# see evalutation result
mse = mean_squared_error(y_test, preds)
print(mse)

roc = roc_auc_score(y_test, preds)
print(roc)

print(classification_report(y_test, preds))
# plot feature importance
rf_feature = {"Importance": model.feature_importances_, "Feature": X_all.columns}
rf_feature = pd.DataFrame(rf_feature)
plt.figure(figsize=(15, 6))
sns.barplot("Feature", "Importance", data = rf_feature)
# parameter check in random forest
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [100, 150, 200, 250, 300, 350, 400]
max_features = ['auto', 'sqrt']

max_depth = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]

min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

scoring = {"prec":"precision", "recal":"recall", "auc":"roc_auc"}

rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring="precision", n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
rf_random.best_params_
# XGBoost
import xgboost as xgb
xgb_model = xgb(n_estimators=200, learning_rate=0.05)
xgb_H = xgb_model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(X_test, y_test)], verbose=True)
xgb_preds = xgb_model.predict(X_test)
print(len(xgb_preds))
# see the evaluation results
print(classification_report(xgb_preds, y_test))
