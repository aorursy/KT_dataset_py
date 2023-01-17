import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data=pd.read_csv("../input/cereals-rating/cereals.csv")
dataset=data.copy()
dataset
dataset.info()
dataset.describe()
plt.figure(figsize=(8,15))

plt.barh(dataset["name"].unique(),(dataset["rating"]))

plt.ylabel("Product Name",fontsize=15)

plt.xlabel("Rating",fontsize=15)

plt.title("Product Rating",fontsize=15)

plt.show()
plt.bar(dataset["type"].unique(),dataset["type"].value_counts(),color="Red")

plt.ylabel("Type count",fontsize=15)

plt.xlabel("Type",fontsize=15)

plt.title("Type count",fontsize=15)

plt.show()
plt.bar(dataset["mfr"].unique(),dataset["mfr"].value_counts(),color="green")

plt.ylabel("count",fontsize=15)

plt.xlabel("MFR",fontsize=15)

plt.title("MFR count",fontsize=15)

plt.show()
import seaborn as sns

df = dataset.corr()

plt.figure(figsize=(12,12))

plt.title('Correlation matrix', fontsize=14)

ax = sns.heatmap(df, annot=True, fmt='.2f',square=True,linewidths=.5)

ax.set_ylim(len(df)-0.1, -0)

plt.show()
dataset.isnull().sum()
dataset.drop(["name"],axis=1,inplace = True)
dataset=pd.get_dummies(dataset,drop_first=True)
dataset
X1=dataset.drop(["rating"],axis=1)
Y=dataset["rating"]
X1
from sklearn.preprocessing import scale
X=scale(X1)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
lr=LinearRegression(fit_intercept=True)

model1=lr.fit(X_train,Y_train)

prediction=lr.predict(X_test)
r2test=model1.score(X_test,Y_test)

r2train=model1.score(X_train,Y_train)

print(r2train,r2test)
rf=RandomForestRegressor()

model2=rf.fit(X_train,Y_train)
r2test=model2.score(X_test,Y_test)

r2train=model2.score(X_train,Y_train)

print(r2train,r2test)
X