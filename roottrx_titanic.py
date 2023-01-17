# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe().T
train_df.info()
def bar_plot(variable):
    """
    input: varibale, ex: "Sex"
    output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    var_value = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9, 3))
    plt.bar(var_value.index, var_value)
    plt.xticks(var_value.index, var_value.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)
    plt.show()
    print(f"{variable}: \n {var_value}")
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)
category2 = ['Cabin', 'Name', 'Ticket']
for c in category2:
    print(f"{train_df[c].value_counts()} \n")
def plot_hist(variable):
    plt.figure(figsize=(9, 3))
    plt.hist(train_df[variable], bins=10)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title(f"{variable} distribution with hist")
    plt.show()

numeric_var = ['Fare', 'Age', 'PassengerId']
for n in numeric_var:
    plot_hist(n)
# Pclass vs Survived
train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index = False).mean().sort_values(by='Survived', ascending=False)
# Sex vs Survived
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index = False).mean().sort_values(by='Survived', ascending=False)
# Sex vs Survived
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index = False).mean().sort_values(by='Survived', ascending=False)
# Sex vs Survived
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index = False).mean().sort_values(by='Survived', ascending=False)
def detect_outliers(df, features):
    outlier_indices = []
    
    for c in features:
        # first quartile
        Q1 = np.percentile(df[c], 25)
        # third quartile
        Q3 = np.percentile(df[c], 75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
train_df.loc[detect_outliers(train_df, ['Age', 'SibSp', 'Parch', 'Fare'])]
# Drop outliers
train_df = train_df.drop(detect_outliers(train_df, ['Age', 'SibSp', 'Parch', 'Fare']), axis=0).reset_index(drop=True)
train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df['Embarked'].isnull()]
train_df.boxplot(column='Fare', by='Embarked')
plt.show()
train_df["Embarked"] = train_df['Embarked'].fillna("C")
train_df[train_df['Fare'].isnull()]
train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass'] == 3]['Fare']))
list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f");
g = sns.factorplot(x = "SibSp", y = 'Survived', data =train_df, kind="bar", size = 9)
g.set_ylabels('Survived Probability')
plt.show()
g = sns.factorplot(x = "Parch", y = 'Survived', kind = "bar", data = train_df, size = 6)
g.set_ylabels("Survived Probability")
plt.show()
g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probability");
g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins=25)
plt.show();
train_df[train_df["Age"].isnull()]
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
age_med = train_df["Age"].median()
train_df["Age"] = train_df["Age"].fillna(age_med)
train_df["Age"].isna().sum()
train_df['Name'].head(10)
train_df.drop("Name", axis=1, inplace=True)
train_df.head()
train_df.drop("Ticket", axis=1, inplace=True)
train_df.head()
train_df.drop("PassengerId", axis=1, inplace=True)
train_df.head()
train_df.drop("Cabin", axis=1, inplace=True)
train_df.head()
train_df = pd.get_dummies(train_df, columns=["Embarked"])
train_df.head()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
test = train_df[train_df_len:]
test.drop(labels=["Survived"], axis = 1, inplace=True)
train = train_df[:train_df_len]
X_train = train.drop(labels = 'Survived', axis=1)
y_train = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)
print("X_test", len(X_test))
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100, 2)
acc_log_test = round(logreg.score(X_test, y_test)*100, 2)
print(f"Training accuracy: % {acc_log_train}")
print(f"Test accuracy: % {acc_log_test}")
results = pd.Series(logreg.predict(test), name='Survived').astype(int)
results.to_csv("titanic.csv", index = False)
