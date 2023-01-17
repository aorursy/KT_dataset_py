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
df = pd.read_csv("../input/titanic_data.csv")

df.head()
df.columns
drop_rows = ["PassengerId", "Name", "Sex", "Ticket", "Cabin", "Embarked"]

cdf = df.drop(drop_rows, axis=1)
cdf.head()
cdf.describe()
cdf.groupby("Survived").mean()
cdf.groupby(df["Age"].isnull()).mean()
import seaborn as sns

import matplotlib.pyplot as plt



for i in ["Age", "Fare"]:

    died = list(cdf[cdf["Survived"]==0][i].dropna())

    survived = list(cdf[cdf["Survived"]==1][i].dropna())

    xmin= min(min(died), min(survived))

    xmax= max(max(died), max(survived))

    width = (xmax-xmin)/40

    sns.distplot(died, color="r", kde=False, bins=np.arange(xmin,xmax,width))

    sns.distplot(survived, color="g", kde=False, bins=np.arange(xmin,xmax,width))

    plt.legend(["Did not Survived", "Survived"])

    plt.title("Overlaid histogram for {}".format(i))

    plt.show()
for i, col in enumerate(["Pclass", "SibSp", "Parch"]):

    plt.figure(i)

    sns.catplot(x=col, y="Survived", data=cdf, kind="point", aspect=2)
cdf["Family"] = cdf["SibSp"] + cdf["Parch"]

cdf.drop(["SibSp", "Parch"], axis=1, inplace=True)

sns.catplot(x="Family", y="Survived", data=cdf, kind="point", aspect=2)
cdf["Age"].fillna(cdf["Age"].mean(), inplace=True)

cdf.isnull().sum()
cdf.head(10)
rows_to_drop = ["PassengerId", "Pclass", "Name", "Age", "SibSp", "Parch", 

                "Fare"]

idf = df.drop(rows_to_drop, axis=1)

idf.head()
idf.info()
idf.groupby(idf["Cabin"].isnull()).mean()
idf["Cabin_ind"] = np.where(idf["Cabin"].isnull(), 0, 1)

idf.head()
for i, col in enumerate(["Cabin_ind", "Sex", "Embarked"]):

    plt.figure(i)

    sns.catplot(x=col, y="Survived", data=idf, kind="point", aspect=2)
idf.pivot_table("Survived", index="Sex", columns="Embarked", aggfunc="count")
idf.pivot_table("Survived", index="Cabin_ind", columns="Embarked", 

                aggfunc="count")
idf.head()
gender_num = {"male": 0, "female": 1}

idf["Sex"] = idf["Sex"].map(gender_num)

idf.head()
idf.drop(["Cabin", "Embarked"], axis=1, inplace=True)

idf.head()
df.drop(["Name", "Ticket", "Embarked"], axis=1, inplace=True)

df.head()
df["Sex"] = idf["Sex"]

df["Age"] = cdf["Age"]

df["Cabin"] = idf["Cabin_ind"]

df.head()
from sklearn.model_selection import train_test_split

x = df.drop("Survived", axis=1)

y = df["Survived"]



x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, random_state=42)

for dataset in [y_train, y_test]:

    print(round(len(dataset)/len(x), 2))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

import warnings 

warnings.filterwarnings("ignore", category=FutureWarning)



rf = RandomForestClassifier()

scores = cross_val_score(rf,x, y, cv=5)

scores
def print_result(result):

    print("BEST PARAMS:{}\n".format(result.best_params_))

    

    means = result.cv_results_["mean_test_score"]

    std = result.cv_results_["std_test_score"]

    for mean, std, params in zip(means, std, result.cv_results_["params"]):

        print("{} (+/-{}) for {}".format(round(mean, 3), round(std, 3), params))
parameters = {

    "n_estimators": [5, 50, 100], 

    "max_depth":[2, 10, 20, None]

}



cv = GridSearchCV(rf, parameters, cv=5)

cv.fit(x, y)

print_result(cv)
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=100, max_depth=20)

rf.fit(x_train,y_train)

y_hat = rf.predict(x_test)

print("The accuracy score: ",accuracy_score(y_test, y_hat))

print("The f1_score:\n", classification_report(y_test, y_hat))