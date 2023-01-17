import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_train.head()
df_train.info()
df_train.describe()
np.round(df_train.isna().sum() / df_train.shape[0], 1)
sns.heatmap(df_train.corr())

plt.show()
print(df_train['Survived'].value_counts())

print(df_train['Pclass'].value_counts())

print(df_train['Sex'].value_counts())

print(df_train['SibSp'].value_counts())

print(df_train['Parch'].value_counts())

print(df_train['Embarked'].value_counts())
plt.subplots(figsize=(4, 4))

sns.countplot(x='Sex', hue='Survived', data=df_train)

plt.show()
plt.subplots(figsize=(4, 4))

sns.countplot(x='Embarked', hue='Survived', data=df_train)

plt.show()
plt.subplots(figsize=(4, 4))

sns.countplot(x='Pclass', hue='Survived', data=df_train)

plt.show()
plt.subplots(figsize=(4, 4))

sns.countplot(x='Parch', hue='Survived', data=df_train)

plt.show()
f, ax = plt.subplots(figsize=(4, 4))

ax = sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'])

ax = sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'])
df_train = df_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

df_train.head()
df_train['Sex'] = df_train['Sex'].replace('male', 0)

df_train['Sex'] = df_train['Sex'].replace('female', 1)

df_train['Sex'].value_counts()
df_train['Embarked'] = df_train['Embarked'].replace('S', 0)

df_train['Embarked'] = df_train['Embarked'].replace('C', 1)

df_train['Embarked'] = df_train['Embarked'].replace('Q', 2)

df_train['Embarked'].value_counts()
df_train = df_train.dropna()
df_train.count()
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import classification_report
X = df_train[[c for c in df_train.columns if c != "Survived"]]

y = df_train["Survived"]

X.head()
y.head()
model_dt = DecisionTreeClassifier()
print("Decision Tree: ")

kf = KFold(n_splits=5, random_state=32, shuffle=True)

dt_score_list = list()

cnt = 0



for train_index, test_index in kf.split(X):

    

    print("Fold {}:".format(cnt + 1))

    cnt += 1

    

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



    model_dt.fit(X_train, y_train)

    y_hat = model_dt.predict(X_test)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_hat)

    

    df_temp = pd.DataFrame()

    df_temp["Precision"] = [prec[0], prec[1]]

    df_temp["Recall"] = [rec[0], rec[1]]

    df_temp["F1-score"] = [f1[0], f1[1]]

    df_temp.index.name = 'Class'

    print(df_temp)

    

    score = {'precision' : df_temp['Precision'].tolist(), 

             'recall' : df_temp['Recall'].tolist(),

             'f1': df_temp['F1-score'].tolist()}

    dt_score_list.append(score)
average_f1 = np.mean([np.mean(ele['f1']) for ele in dt_score_list])

print("Average f-score of Decision Tree: {:.2f}".format(average_f1))
model_mlp = MLPClassifier(hidden_layer_sizes=(150,100,50),

                          max_iter=300,

                          activation = 'relu',

                          solver='sgd',

                          random_state=1)
print("Multi-layer Perceptron: ")

kf = KFold(n_splits=5, random_state=32, shuffle=True)

mlp_score_list = list()

cnt = 0



for train_index, test_index in kf.split(X):

    

    print("Fold {}:".format(cnt + 1))

    cnt += 1

    

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



    model_mlp.fit(X_train, y_train)

    y_hat = model_mlp.predict(X_test)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_hat)

    

    df_temp = pd.DataFrame()

    df_temp["Precision"] = [prec[0], prec[1]]

    df_temp["Recall"] = [rec[0], rec[1]]

    df_temp["F1-score"] = [f1[0], f1[1]]

    df_temp.index.name = 'Class'

    print(df_temp)

    

    score = {'precision' : df_temp['Precision'].tolist(), 

             'recall' : df_temp['Recall'].tolist(),

             'f1': df_temp['F1-score'].tolist()}

    mlp_score_list.append(score)
average_f1 = np.mean([np.mean(ele['f1']) for ele in mlp_score_list])

print("Average f-score of Multi-layer Perceptron: {:.2f}".format(average_f1))
model_gnb = GaussianNB()
print("Decision Tree: ")

kf = KFold(n_splits=5, random_state=32, shuffle=True)

gnb_score_list = list()

cnt = 0



for train_index, test_index in kf.split(X):

    

    print("Fold {}:".format(cnt + 1))

    cnt += 1

    

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



    model_gnb.fit(X_train, y_train)

    y_hat = model_gnb.predict(X_test)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_hat)

    

    df_temp = pd.DataFrame()

    df_temp["Precision"] = [prec[0], prec[1]]

    df_temp["Recall"] = [rec[0], rec[1]]

    df_temp["F1-score"] = [f1[0], f1[1]]

    df_temp.index.name = 'Class'

    print(df_temp)

    

    score = {'precision' : df_temp['Precision'].tolist(), 

             'recall' : df_temp['Recall'].tolist(),

             'f1': df_temp['F1-score'].tolist()}

    gnb_score_list.append(score)
average_f1 = np.mean([np.mean(ele['f1']) for ele in gnb_score_list])

print("Average f-score of Gaussian Naive Bayes: {:.2f}".format(average_f1))