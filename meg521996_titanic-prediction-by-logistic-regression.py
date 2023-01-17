# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


%matplotlib inline                 



import pandas as pd

import numpy as np                 

import matplotlib.pyplot as plt    

import seaborn as sns              





from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
df.shape
df.info()
df.describe()
df.Age
df["Embarked"].value_counts()
df["Pclass"].value_counts()
df["Cabin"].value_counts()
missing_df = df.isnull()

missing_df.head()
missing_df["Embarked"].value_counts()
col_names = df.columns.values.tolist()

col_names
col_names = df.columns.values.tolist()

for column in col_names:

    print(column)

    print(missing_df[column].value_counts())

    print("")
count_df = pd.isnull(df).sum()

count_df
df["Embarked"]=df["Embarked"].fillna("S")
df["Cabin"].dropna()
df["Survived"].hist(bins=2,rwidth=0.90,grid=False)
g = sns.factorplot("Embarked", "Survived", data=df, kind="bar", legend=True)
df.hist(bins=10,figsize=(9,7),grid=False)
g = sns.factorplot("Survived","Age","Sex", data=df, kind="bar", legend=True)
g = sns.factorplot("Pclass", "Survived", "Sex", data=df, kind="bar", size=5, palette="muted", legend=False)
g = sns.FacetGrid(data=df, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age", color="red")
g = sns.FacetGrid(data=df, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
ax = sns.boxplot(x="Survived", y="Age", 

                data=df)

ax = sns.stripplot(x="Survived", y="Age",

                   data=df, jitter=True,

                   edgecolor="black")

sns.plt.title("Survival no. by Age",fontsize=10)
def fill_missing_fare(df):

    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()

    df["Fare"] = df["Fare"].fillna(median_fare)

    return df

df=fill_missing_fare(df)
df["Fare"]=df.Fare.fillna(df["Fare"].mean(), inplace=True)
df["Age"]=df.Age.fillna(df["Age"].mean(), inplace=True)

df["Age"].head()
df["FamSize"] = df["SibSp"] + df["Parch"] + 1
df["FamSize"].value_counts()
df.columns
from sklearn.preprocessing import LabelEncoder



le_sex = LabelEncoder()

le_sex.fit(df["Sex"])



encoded_sex = le_sex.transform(df["Sex"])

df["Sex"] = encoded_sex



le_embarked = LabelEncoder()

le_embarked.fit(df["Embarked"])



encoded_embarked = le_embarked.transform(df["Embarked"])

df["Embarked"] = encoded_embarked
X = df[['Pclass', 'Sex', 'SibSp','Parch', 'Embarked', 'FamSize']].values

X[0:5]
y = df['Survived'].values

y[0:5]
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(int))

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

LR
ypred = LR.predict(X_test)

ypred
from sklearn.metrics import jaccard_score

jaccard_score(y_test, ypred)
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, LR.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, ypred))
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))