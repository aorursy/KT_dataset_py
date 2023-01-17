import pandas as pd

import seaborn as sns

import numpy as np
d = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
d.isnull().sum()
d.info()
df = d.copy()
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()

for i in df.columns:

    df[i] = l.fit_transform(df[i])
df.corrwith(df["class"])
df = df.drop(columns="veil-type",axis=1)
sns.countplot(df["class"])
x = df.drop(columns="class",axis=1)

y = df["class"]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test,y_pred))

print("accuracy is (%) =",accuracy_score(y_test,y_pred)*100)