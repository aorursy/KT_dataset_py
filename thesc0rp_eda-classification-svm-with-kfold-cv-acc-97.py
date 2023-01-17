print("Hello World! This is my first submission. Let's begin!")
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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
dataset = pd.read_csv('/kaggle/input/drug-classification/drug200.csv')
dataset
dataset.dtypes
dataset.isnull().sum()
plt.figure(figsize=(8,6))
dcount = sns.countplot('Drug', data=dataset)
plt.figure(figsize=(8,6))
dcount = sns.countplot('Drug', hue='Sex', data=dataset)
plt.figure(figsize=(8,6))
ax = sns.boxplot('Sex', 'Age', data=dataset).set(ylim=(0, 80))
sexcnt = sns.countplot('Sex', data=dataset).set_yticks([i*10 for i in range(12)])
agevdrug = sns.catplot('Drug', 'Age', data=dataset)
fig, ax = plt.subplots(1,2, figsize=(18, 7))
agevbp = sns.violinplot('BP', 'Age', data=dataset, hue = 'Sex', ax = ax[0]).set(ylim=(0,100))
agevch = sns.violinplot('Cholesterol', 'Age', data=dataset, hue = 'Sex', ax = ax[1]).set(ylim=(0,100))
fig, ax = plt.subplots(1, 3, figsize=(22,5))
hbp = sns.countplot(dataset[dataset['BP'] == 'HIGH'].iloc[:, -1], ax=ax[0],order=list(dataset['Drug'].unique())).set(ylim=(0,40), xlabel='HIGH BP')
nbp = sns.countplot(dataset[dataset['BP'] == 'NORMAL'].iloc[:, -1], ax=ax[1],order=list(dataset['Drug'].unique())).set(ylim=(0,40), xlabel='NORMAL BP')
lbp = sns.countplot(dataset[dataset['BP'] == 'LOW'].iloc[:, -1], ax=ax[2],order=list(dataset['Drug'].unique())).set(ylim=(0,40), xlabel='LOW BP')
plt.figure(figsize=(8,6))
chcount = sns.countplot('Drug', hue='Cholesterol', data=dataset)
plt.figure(figsize=(8,6))
nakvsbp = sns.boxplot('BP', 'Na_to_K', data = dataset).set(ylim=(0,40))
nakvsdrug = sns.catplot('Drug', 'Na_to_K', data=dataset)
X = dataset.iloc[: :-1]
y = dataset.iloc[:, -1].values
ct = ColumnTransformer([
    ("onehot", OneHotEncoder(), [1]), 
    ("ordBP", OrdinalEncoder(categories = [['HIGH', 'NORMAL', 'LOW']]), [2]),
    ("ordChol", OrdinalEncoder(categories = [['HIGH', 'NORMAL']]), [3]),
    ("stdscl", StandardScaler(), [0, 4])], n_jobs = -1)
X = ct.fit_transform(X)[::-1]
le = LabelEncoder().fit(y)
le.classes_
y = le.transform(y)
model = SVC()

num_splits = 5
kfold = KFold(num_splits)
train_accs, test_accs = [], []
for train_index, test_index in kfold.split(X):    
    # Splitting the data into train and test set 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Training the model
    model.fit(X_train, y_train)
    
    # Predicting the results on the training set and the test set
    train_accs.append(accuracy_score(y_train, model.predict(X_train)) * 100)
    test_accs.append(accuracy_score(y_test, model.predict(X_test)) * 100)
print("\tTraining \t Test")
for i in range(1, num_splits+1):
    print(i, "\t", train_accs[i-1], "\t", test_accs[i-1])
print("Average training set accuracy: {:.2f}".format(sum(train_accs) / num_splits))
print("Average test set accuracy: {:.2f}".format(sum(test_accs) / num_splits))