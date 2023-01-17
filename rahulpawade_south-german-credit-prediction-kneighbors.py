import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/south-german-credit-prediction/train.csv")
train.info()
train.head()
train.shape
plt.figure(figsize=(25,25))

sns.heatmap(train.corr(),annot=True)
for i in train.columns:

    print(train[i].value_counts())
sns.countplot(train["kredit"])
test = pd.read_csv("../input/south-german-credit-prediction/test.csv")
test.shape
test.info()
test.head()
test.describe()
for i in test.columns:

    print(test[i].value_counts())
x_train = train.drop(columns=["kredit","Id"],axis=1)

y_train = train["kredit"]
x_test = test.drop(columns="Id",axis=1)
from sklearn.preprocessing import StandardScaler

s = StandardScaler()

x_train = s.fit_transform(x_train)

x_test = s.fit_transform(x_test)
x_train.shape,x_test.shape,y_train.shape
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
d = pd.read_csv("../input/south-german-credit-prediction/SampleSubmission.csv")
s = {"Id":d["Id"],"kredit":y_pred}

s = pd.DataFrame(s)

s.to_csv("submission.csv",index=False)
s.head()