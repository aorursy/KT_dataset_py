# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import plotly.express as px #data visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/titanic/train.csv")
train_df.head()
train_df.describe()
train_df.isnull().sum()
survived = train_df.groupby("Survived")["PassengerId"].count().reset_index(name="count")
survived_pie = px.pie(survived, values="count", names="Survived", title="Survivor Spread")
survived_pie.show()
gender = train_df.groupby("Sex")["PassengerId"].count().reset_index(name="count")
gender_pie = px.pie(gender, values="count", names="Sex", title="Gender Spread")
gender_pie.show()
male_df = train_df[train_df.Sex == "male"]
male = male_df.groupby("Survived")["PassengerId"].count().reset_index(name="count")
male_pie = px.pie(male, values="count", names="Survived", title="Male Survival Spread", color="Survived", color_discrete_map={1:"lightcyan", 0:"royalblue"})
male_pie.show()
female_df = train_df[train_df.Sex == "female"]
female = female_df.groupby("Survived")["PassengerId"].count().reset_index(name="count")
female_pie = px.pie(female, values="count", names="Survived", title="Female Survival Spread", color="Survived", color_discrete_map={1:"lightcyan", 0:"royalblue"})
female_pie.show()
pclass_1_df = train_df[train_df.Pclass == 1]
pclass_1 = pclass_1_df.groupby("Survived")["PassengerId"].count().reset_index(name="count")
pclass_1_pie = px.pie(pclass_1, values="count", names="Survived", title="Pclass1 Survived Spread", color="Survived", color_discrete_map={1:"#09FF00", 0:"#FF0000"})
pclass_1_pie.show()
pclass_2_df = train_df[train_df.Pclass == 2]
pclass_2 = pclass_2_df.groupby("Survived")["PassengerId"].count().reset_index(name="count")
pclass_2_pie = px.pie(pclass_2, values="count", names="Survived", title="Pclass2 Survived Spread", color="Survived", color_discrete_map={1:"#09FF00", 0:"#FF0000"})
pclass_2_pie.show()
pclass_3_df = train_df[train_df.Pclass == 3]
pclass_3 = pclass_3_df.groupby("Survived")["PassengerId"].count().reset_index(name="count")
pclass_3_pie = px.pie(pclass_3, values="count", names="Survived", title="Pclass3 Survived Spread", color="Survived", color_discrete_map={1:"#09FF00", 0:"#FF0000"})
pclass_3_pie.show()
age_survivor_df = train_df[["Survived", "Age"]]
age_survivor_df = age_survivor_df.dropna()
age_survivor_df.groupby(["Survived"]).mean()
age_survivor_1_df = age_survivor_df[age_survivor_df.Survived == 1]
age_survivor_1_df = age_survivor_1_df.drop(columns=["Survived"])
age_survivor_1_df.hist()
plt.title("Histogram for Age of Survivors")
age_survivor_0_df = age_survivor_df[age_survivor_df.Survived == 0]
age_survivor_0_df = age_survivor_0_df.drop(columns=["Survived"])
age_survivor_0_df.hist()
plt.title("Histogram for Age of Non-survivors")

train_df = train_df[train_df["Age"].notna()]
train_df = train_df.replace(["male"],0)
train_df = train_df.replace(["female"],1)
X_train = train_df[["Sex", "Pclass", "Age"]]
Y_train = train_df["Survived"]
mean_age = train_df["Age"].mean()
test_df = pd.read_csv("../input/titanic/test.csv")
X_test = test_df[["Sex", "Pclass", "Age"]]
X_test = X_test.replace(["male"],0)
X_test = X_test.replace(["female"],1)
X_test = X_test.fillna(mean_age)
X_train.shape, Y_train.shape, X_test.shape

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_test)
print(neigh.predict([[1, 3, 5]])) #Prediction for Sex: female, Pclass: 3, Age: 5
print(neigh.predict([[0, 1, 10]])) #Prediction for Sex: male, Pclass: 1, Age: 10
print(neigh.predict([[0, 1, 70]])) #Prediction for Sex: male, Pclass: 1, Age: 70
print(neigh.predict([[1, 3, 45]])) #Prediction for Sex: female, Pclass: 3, Age: 45

#Saving our predicted Y values
Y_pred = neigh.predict(X_test)
print(Y_pred.shape)
knn_accuracy = neigh.score(X_train, Y_train)
print(knn_accuracy)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv("submission.csv",index=False)

