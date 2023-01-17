import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
df = pd.read_csv("/kaggle/input/hr-analytics/HR_comma_sep.csv")

df.head()
df.describe()
df.isnull().sum()
df.corr()
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, annot=True, linewidth=1.5, cmap="YlGnBu")
sns.barplot(x = 'left', y = 'salary', data = df)
sns.barplot(x = 'left', y = 'Department', data = df)
y = df["left"]
df = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
salary_dummies = pd.get_dummies(df.salary, prefix="salary")

df_with_dummies = pd.concat([df,salary_dummies],axis='columns')

df_with_dummies.head()

df_with_dummies.drop('salary',axis='columns',inplace=True)



df_with_dummies.head()
X = df_with_dummies
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)
model = LogisticRegression()

model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test,y_test)
