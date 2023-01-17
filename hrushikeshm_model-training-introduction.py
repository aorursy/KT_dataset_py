!pip install seaborn==0.11
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import imp
import os

import IPython

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import seaborn as sns

imp.reload(sns)
print("Versions:\nPandas(expected: 1.1.2):", pd.__version__, "\nSeaborn(expected: 0.11.0):", sns.__version__)

from matplotlib import pyplot as plt

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
print("Data files:")
base_dir="/kaggle/input/titanic"
for dirname, _, filenames in os.walk(base_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create 
# a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!head /kaggle/input/titanic/train.csv
train_df = pd.read_csv(base_dir + "/train.csv")
print("Train sample")
train_df.sample(5)
test_df = pd.read_csv(base_dir + "/test.csv")
print("Test sample")
test_df.sample(5)
#Basic info on data
print("Training data Info:\n","-"*20, sep="")
train_df.info()
print("\nTraining data basic stats:\n","-"*25, sep="")
print(train_df.describe(include="all"))
print("\nNull values training data:\n","-"*25, sep="")
print(train_df.isnull().sum())
print("\nNull values test data:\n","-"*25, sep="")
print(test_df.isnull().sum())
# Substitue null values with Median for "numeric", Mode/"Missing" for "categorical" values:
dfs_to_clean = [train_df, test_df]
for df in dfs_to_clean:
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)    
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    # Filling cabin 
    df["Cabin"].fillna("Missing", inplace=True)

print("\nNull values training data:\n","-"*25, sep="")
print(train_df.isnull().sum())
print("\nNull values test data:\n","-"*25, sep="")
print(test_df.isnull().sum())

train_df.drop(["PassengerId", "Ticket"], axis=1, inplace = True)
train_df.sample(5)
for df in dfs_to_clean:    
    df["FamilySize"] = 1 + df["SibSp"] + df["Parch"]
    df["IsAlone"] = (df["FamilySize"]<=1).astype(int)
    df["NamePrefix"] = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

# Replace rare name prefixes with "Misc"
np_value_counts = train_df["NamePrefix"].value_counts()
print("prefixes before cleanup:\n", np_value_counts, sep="")
name_prefixes = set(np_value_counts[np_value_counts >= 10].index.to_list())
for df in dfs_to_clean:
    df["NamePrefix"] = df["NamePrefix"].apply(lambda x: x if x in name_prefixes else "Misc")

print("\nprefixes after cleanup:\n", train_df["NamePrefix"].value_counts(), sep="")
train_df.sample(5)
sns.heatmap(train_df.corr().abs(), annot=True, cmap="YlGnBu")
target="Survived"
numerical=["Pclass", "Fare", "Age", "SibSp", "Parch", "FamilySize"]
categorical=["IsAlone", "Sex", "NamePrefix", "Embarked", "Cabin"]
for col in numerical:
    plt.figure()
    ax=sns.displot(train_df, x=col, hue=target)

num_people = len(train_df)
for col in categorical:
    plt.figure()
    ax = sns.countplot(data=train_df[[col, target]], x=col, hue=target)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:.0f}%'.format(height/num_people*100),
                ha="center") 
    plt.show()

train_df["Cabin"].value_counts()
# Encoding categorical features
ohe_cols = ["Sex", "NamePrefix", "Embarked"]
dummy_df = None 
for df in dfs_to_clean:
    dummy_df = pd.get_dummies(df[ohe_cols], prefix=ohe_cols)
#     print(dummy_df.sample(2))
    for col in dummy_df.columns:
        df[col] = dummy_df[col]
dummy_cols = dummy_df.columns.to_list()
train_cols = ["IsAlone"] + numerical + dummy_cols  
print(train_df[train_cols].sample(3))
from sklearn import model_selection, linear_model, svm, tree, metrics
# Cross Validate
header=["Classifier", "Training Time", "Accuracy", "Precision", "Recall", "F1"]
scoring_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

print("x +/- y Represents metric value x, with 95% of values lying within +/- y") # Confidence intervals
def record_metric(s):
    return "%0.2f +/- %0.2f" % (s.mean(), s.std() * 2)

data = []
score = None
for clf in [linear_model.LogisticRegression(), svm.SVC(), tree.DecisionTreeClassifier()]:
    score = model_selection.cross_validate(
        clf, train_df[train_cols], train_df[target], 
        scoring=scoring_cols)
    row = [type(clf).__name__, record_metric(score["fit_time"])]
    for metric in scoring_cols:
        row.append(record_metric(score["test_"+metric]))
    data.append(row)
print(score)
pd.DataFrame(data, columns=header)
print("Accuracy on test data")
for clf in [linear_model.LogisticRegression(), svm.SVC(), tree.DecisionTreeClassifier()]:
    clf.fit(train_df[train_cols], train_df[target])
    # optionally pickle model for future use
    y_pred = clf.predict(test_df[train_cols])
    print(y_pred[:5])
#     print("%s - %.2f"%(type(clf).__name__, metrics.accuracy_score(test_df[target], y_pred)))
test_df.isnull().sum()
