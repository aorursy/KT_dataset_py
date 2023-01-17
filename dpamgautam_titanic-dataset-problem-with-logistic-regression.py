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

%matplotlib inline
training_data = pd.read_csv("../input/train.csv")

testing_data = pd.read_csv("../input/test.csv")
(training_data.head(5))
training_data.tail(5)
sns.heatmap(training_data.isnull())
sns.countplot(x="Survived", data=training_data)
sns.countplot(x="Sex", data=training_data)
sns.countplot(x="Survived", hue="Sex", data=training_data)
sns.distplot(training_data["Age"].dropna())
sns.boxplot(x="Pclass", y="Age", data=training_data)
def impute_age(cols):

    age = cols[0]

    pclass = cols[1]

    if pd.isnull(age):

        if pclass == 1:

            return 37

        elif pclass == 2:

            return 29

        else:

            return 24

    else:

        return age
training_data["Age"] = training_data[["Age","Pclass"]].apply(impute_age, axis=1)
sns.heatmap(data = training_data.isnull())
training_data.drop("Cabin", axis=1, inplace=True)
training_data.head(5)
training_data.info()
sex = pd.get_dummies(training_data["Sex"], drop_first=True)

embark = pd.get_dummies(training_data["Embarked"], drop_first=True)
training_data.drop(["Sex","Embarked","Name","Ticket"], axis=1, inplace=True)
training_data = pd.concat([training_data, sex, embark], axis=1)
training_data.head(5)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_data.drop("Survived", axis=1), training_data["Survived"], test_size = 0.3, random_state = 101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
pred = logmodel.predict(X_test)

pred
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))