import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
train = pd.read_csv("../input/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

test = pd.read_csv("../input/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train = train.dropna()
train.head()
train.iloc[:,1:16].hist(figsize =(10,10),bins=50)
plt.show()
train["workclass"].value_counts().plot(title="workclass", kind='pie', label='')
plt.show()
train["education"].value_counts().plot(title="education", kind='pie', label='')
plt.show()
train["marital.status"].value_counts().plot(title="marital.status", kind='pie', label='')
plt.show()
train["occupation"].value_counts().plot(title="occupation", kind='pie', label='')
plt.show()
train["relationship"].value_counts().plot(title="relationship", kind='pie', label='')
plt.show()
train["race"].value_counts().plot(title="race", kind='pie', label='')
plt.show()
train["sex"].value_counts().plot(title="sex", kind='pie', label='')
plt.show()
train["native.country"].value_counts().plot(title="native.country", kind='pie', label='')
plt.show()
Xtrain =train[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Ytrain=train["income"]
knn = KNeighborsClassifier(n_neighbors=20)
scoresknn = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scoresknn.mean()
gbc= GradientBoostingClassifier(max_depth=2,learning_rate=1)
scoresgbc=cross_val_score(gbc,Xtrain,Ytrain,cv=10)
scoresgbc.mean()
n = GaussianNB()
scoresnb = cross_val_score(n, Xtrain, Ytrain, cv=10)
scoresnb.mean()