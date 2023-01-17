# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report
df = pd.read_csv("../input/kyphosis.csv")
df.head()
df.isnull().sum().sum()
sns.countplot(df['Kyphosis'])
sns.distplot(df['Age'], bins=30)
sns.heatmap(df.corr(), annot=True)
sns.boxplot('Kyphosis', 'Age', data=df)
df['Start'].nunique()
sns.boxplot('Kyphosis', 'Start', data=df)
sns.boxplot('Kyphosis', 'Number', data=df)
X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis'].map({'absent':0, 'present':1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)
train_mean = X_train.mean()
train_std = X_train.std()
X_train = (X_train - train_mean)/train_std
X_test = (X_test - train_mean)/train_std
y_test.value_counts()
models = {'Logistic Regression': LogisticRegression(class_weight='balanced'),
          'KNN': KNeighborsClassifier(n_neighbors=1)}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    #print(model_name, f1_score(y_test, predict))
    print(model_name)
    print(classification_report(y_test, predict))
    print("====================\n")
