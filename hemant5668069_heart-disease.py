import numpy as np

import pandas as pd
df=pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head(10)
print(df.isnull().sum())

print("\n")

print(df.size)

print(df.shape)
print(sum(df.target==1),((sum(df.target==0))/303)*100)
df.corr("pearson")
import seaborn as sns

sns.heatmap(df.corr("pearson"))
x=df.iloc[ : ,0:13]

y=df.iloc[: ,13:]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
classifier=[[ "RandomForestClassifier:",RandomForestClassifier()],["DecisionTreeClassifier :",DecisionTreeClassifier()],["LogisticRegression :",LogisticRegression()],["XGBClassifier :",XGBClassifier()]]

for name,model in classifier:

  model.fit(x_train,y_train)

  pred=model.predict(x_test)

  print(name,accuracy_score(y_test,pred))

  