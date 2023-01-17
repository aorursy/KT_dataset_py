import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.info()
df.head()
df.describe()
sns.heatmap(df.isna(),yticklabels=0)
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(),annot=True)
sns.pairplot(df)
sns.countplot(df["target"])
from sklearn.preprocessing import StandardScaler

s = StandardScaler()
x = df.drop(columns="target",axis=1)

y = df.target
x = s.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.neighbors import KNeighborsClassifier

for i in range(1,21):

     model = KNeighborsClassifier(n_neighbors=i)

     model.fit(x_train,y_train)

     y_pred = model.predict(x_test)

     print("if k is",i)

     print("accuracy:",accuracy_score(y_test,y_pred))
model = KNeighborsClassifier(n_neighbors=6)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("accuracy is",accuracy_score(y_test,y_pred))
classification_report(y_test,y_pred)
confusion_matrix(y_test,y_pred)
f1_score(y_test,y_pred)