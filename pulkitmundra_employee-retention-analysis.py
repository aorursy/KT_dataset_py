import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt # data visualisation

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/hr-analytics/HR_comma_sep.csv")
df.shape
df.head()
df['left'].value_counts()
df_left = df.loc[df["left"] > 0]

df_left.shape
df_stayed = df.loc[df["left"] == 0]

df_stayed.shape
df_salary=df_left['salary'].value_counts()

df_salary=pd.DataFrame(df_salary)

df_salary.columns = ["frequency_left"]

stayed=df_stayed['salary'].value_counts()

stayed=pd.DataFrame(stayed)

df_salary["frequency_stayed"]=stayed["salary"]

df_salary
df_salary.plot.bar(title="Employees Left vs Employees Stayed by Salary")
df_dep=df_left['Department'].value_counts()

df_dep=pd.DataFrame(df_dep)

df_dep.columns = ["frequency_left"]

dep_stayed=df_stayed['Department'].value_counts()

dep_stayed=pd.DataFrame(dep_stayed)

df_dep["frequency_stayed"]=dep_stayed["Department"]

df_dep
df_dep.plot.bar(title="Employees Left vs Employees Stayed by Department")
df.info()
df["Department"] = df["Department"].astype('category').cat.codes

df["salary"] = df["salary"].astype('category').cat.codes
df.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
x = df.drop(["left"], axis = 1)

x = x.drop(["last_evaluation"], axis = 1)

y = df[["left"]].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=27)
logReg = LogisticRegression()
logReg.fit(x_train, y_train)
predictions = logReg.predict(x_test)
score = logReg.score(x_test, y_test)

print(score)