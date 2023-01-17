# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/titanic/train.csv")

df.head()
df.info()
df.Age.unique()
sns.countplot(x="Survived", data=df)
df.isna().sum()
df.isna().sum().plot(kind="bar")
df.drop(columns=['Cabin'], inplace=True)
without_nullage = df.dropna(axis=0, subset=["Age"])
without_nullage.groupby(by=['Survived']).mean()["Age"]
survived_age = without_nullage.groupby(by=['Survived']).mean()["Age"][1]

not_survived_age = without_nullage.groupby(by=['Survived']).mean()["Age"][0]
df_isnull = df.Age.isna()==True

df_notsurvived = df.Survived==0

df_survived = df.Survived==1
index_list_survived_null = df[df_isnull & df_survived].fillna(survived_age).index

index_list_not_survived_null = df[df_isnull & df_notsurvived].fillna(not_survived_age).index
df["Age"].iloc[index_list_survived_null] = survived_age

df["Age"].iloc[index_list_not_survived_null] = not_survived_age
df.isna().sum()
df.groupby("Embarked").PassengerId.count().plot(kind="bar")
df.Embarked.fillna("S", inplace=True)
df.isna().sum()
df.head()
pclass_1 = df[df.Pclass==1].groupby("Survived").PassengerId.count()
pclass_2 = df[df.Pclass==2].groupby("Survived").PassengerId.count()
pclass_3 = df[df.Pclass==3].groupby("Survived").PassengerId.count()
fig1, ax1 = plt.subplots()

ax1.pie(pclass_1, labels=['not survived','survived'], autopct='%1.2f%%')

plt.title("Pclass 1")

plt.show()
fig2, ax2 = plt.subplots()

ax2.pie(pclass_2, labels=['not survived','survived'], autopct='%1.2f%%')

plt.title("Pclass 2")

plt.show()
fig3, ax3 = plt.subplots()

ax3.pie(pclass_3, labels=['not survived','survived'], autopct='%1.2f%%')

plt.title("Pclass 3")

plt.show()
male = df[df.Sex=="male"].groupby("Survived").PassengerId.count()
female = df[df.Sex=="female"].groupby("Survived").PassengerId.count()
fig4, ax4 = plt.subplots()

ax4.pie(male, labels=['not survived','survived'], autopct='%1.2f%%')

plt.title("Male")

plt.show()
fig5, ax5 = plt.subplots()

ax5.pie(female, labels=['not survived','survived'], autopct='%1.2f%%')

plt.title("Female")

plt.show()
survived_age_df=df[df.Survived==1][["Survived","Age"]]

not_survived_age_df=df[df.Survived==0][["Survived","Age"]]
fig6, ax6 = plt.subplots()

ax6.bar(["survived","not_survived"],[survived_age_df["Age"].mean(),not_survived_age_df["Age"].mean()])

ax6.set_ylabel("Age Means")

plt.show()
survived_fare_df = df[df.Survived==1][["Survived","Fare"]]

not_survived_fare_df = df[df.Survived==0][["Survived","Fare"]]
not_survived_fare_df.max()
survived_fare_df.max()
fig7, ax7 = plt.subplots()

ax7.bar(["survived","not_survived"],[survived_fare_df["Fare"].mean(),not_survived_fare_df["Fare"].mean()])

ax7.set_ylabel("Fare Means")

plt.show()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"]) # 1 -> male, 0 -> female
df.head()
list(df.Embarked.unique())
encoder.fit_transform(list(df.Embarked.unique()))
df.Embarked = encoder.fit_transform(df.Embarked)
df.head()
model_df = df.drop(["Name","Ticket"], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
X = model_df.drop("Survived", axis=1).values

y = model_df["Survived"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
model_logreg = LogisticRegression()

model_logreg.fit(X_train, y_train)

y_preds_logreg = model_logreg.predict(X_test)
logreg_acc = accuracy_score(y_test, y_preds_logreg)

logreg_acc
plot_confusion_matrix(model_logreg, X_test, y_test, normalize="true", cmap="Greys")

plt.title("Normalized Confusion Matrix\nLogisticRegression")

plt.show()
from sklearn.naive_bayes import GaussianNB
model_gauss = GaussianNB()

model_gauss.fit(X_train, y_train)

y_preds_gauss = model_gauss.predict(X_test)
gauss_acc = accuracy_score(y_test, y_preds_gauss)

gauss_acc
plot_confusion_matrix(model_gauss, X_test, y_test, normalize="true", cmap="Greys")

plt.title("Normalized Confusion Matrix\nGaussianNB")

plt.show()
from sklearn.ensemble import RandomForestClassifier
model_randtree = RandomForestClassifier()

model_randtree.fit(X_train, y_train)

y_preds_randtree = model_randtree.predict(X_test)
randtree_acc = accuracy_score(y_test, y_preds_randtree)

randtree_acc
plot_confusion_matrix(model_randtree, X_test, y_test, normalize="true", cmap="Greys")

plt.title("Normalized Confusion Matrix\nRandomForestClassifier")

plt.show()
from sklearn.neural_network import MLPClassifier
model_mlp = MLPClassifier()

model_mlp.fit(X_train, y_train)

y_preds_mlp = model_mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, y_preds_mlp)

mlp_acc
plot_confusion_matrix(model_mlp, X_test, y_test, normalize="true", cmap="Greys")

plt.title("Normalized Confusion Matrix\nMLPClassifier")

plt.show()
from sklearn.neighbors import KNeighborsClassifier
model_kneigh = KNeighborsClassifier()

model_kneigh.fit(X_train, y_train)

y_preds_kneigh = model_kneigh.predict(X_test)
kneigh_acc = accuracy_score(y_test, y_preds_kneigh)

kneigh_acc
plot_confusion_matrix(model_kneigh, X_test, y_test, normalize="true", cmap="Greys")

plt.title("Normalized Confusion Matrix\nKNeighborsClassifier")

plt.show()
compare_dict = {"Model Name": ["LogisticRegression", "GaussianNB", "RandomForestClassifier", "MLPClassifier", "KNeighborsClassifier"], 

                "Accuracy": [logreg_acc, gauss_acc, randtree_acc, mlp_acc, kneigh_acc]}

compare_df = pd.DataFrame(compare_dict)

compare_df.sort_values(by=["Accuracy"], ascending=False)