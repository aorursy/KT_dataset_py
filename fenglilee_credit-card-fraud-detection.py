# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 30)

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/creditcard.csv")
df = df.drop(columns=["Time"])
df

columns = df.columns
print(columns)
print(df.shape)
class_counts = pd.value_counts(df['Class'], sort=True)
class_counts = pd.DataFrame(class_counts).reset_index()
print(class_counts)
f, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='index', y='Class', data=class_counts, ax=ax)

corr = abs(df.corr())
corr = corr.nlargest(n=10, columns='Class')
columns = list(corr.index)
print(columns)
corr = corr[corr.index]
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=corr.columns, xticklabels=corr.index, ax=ax)

target = 'Class'
variables = [item for item in columns if item != 'Class']
variables = np.reshape(variables, [3, 3])
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
for i in range(3):
    for j in range(3):
        sns.boxplot(x=target, y=variables[i, j], data=df, ax=axes[i, j])

from scipy.stats import norm
from scipy import stats
sns.distplot(df['V17'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['V17'], plot=plt)
from collections import Counter
features = df.columns
features = list(features)
features.remove("Class")
features
X, y = df[features], df['Class']
Counter(y)
from imblearn.over_sampling import SMOTE
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X, y)
Counter(y_resampled_smote)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled_smote, y_resampled_smote,test_size=0.3, random_state=0)
Counter(y_test)
from sklearn.linear_model import LogisticRegression
c_param_range = [0.01,0.1,1,10,100]
for c in c_param_range:
    logistic = LogisticRegression(C=c)   
    logistic.fit(X_train, y_train)
    score = logistic.score(X_test, y_test)
    print('score at c=%s is %s' %(c, score, ))
from sklearn.metrics import confusion_matrix
logistic = LogisticRegression(C=10)   
logistic.fit(X_train, y_train)
predict_test = logistic.predict(X_test)
c_matrix = confusion_matrix(y_test, predict_test)
c_matrix