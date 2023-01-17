# Importing essentials

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
ab = pd.read_csv("../input/abalone-dataset/abalone.csv")
ab.head()
ab.info()
ab.describe()
#Adding the Age column and dropping rings column

ab["Age"] = ab["Rings"] + 1.5

ab = ab.drop("Rings",axis=1)

ab.head()
# Visualizing the age

plt.figure(figsize=(12,6))

sns.boxplot(ab.Age)
ab["Sex"].value_counts().plot(kind = 'bar', table = True,figsize=(10,5),xticks = [])
plt.figure(figsize=(10,5))

sns.boxplot(x = "Sex", y = "Age",data=ab)
def one_encode(data,attr):

    dummies = pd.get_dummies(data[attr])

    row = pd.concat([data, dummies], axis=1)

    row = row.drop(attr,axis=1)

    return(row)
ab_one = one_encode(ab,"Sex")
ab_one.dtypes
# Spliting the dataset

from sklearn.model_selection import train_test_split



train_ab, test_ab = train_test_split(ab_one, test_size=0.2, random_state=42)
plt.figure(figsize=(5,10))

plt.title("Correlation matrix")



corr = train_ab.corr()

heat = sns.heatmap(corr[["Age"]],annot= True)
train_labels = train_ab["Age"]

test_labels = test_ab["Age"]



train_ab = train_ab.drop("Age",axis=1)

test_ab =  test_ab.drop("Age",axis=1)
from sklearn.svm import SVR



svr_ab = SVR(kernel="rbf", degree=3, C=100, epsilon=0.1)

svr_ab.fit(train_ab, train_labels)
from sklearn.metrics import mean_squared_error



predictions = svr_ab.predict(train_ab)

svr_mse = mean_squared_error(train_labels, predictions)

svr_rmse = np.sqrt(svr_mse)

print(f"The Root Mean Square Error is {svr_rmse:.2f}")
predictions = svr_ab.predict(test_ab)

svr_mse = mean_squared_error(test_labels, predictions)

svr_rmse = np.sqrt(svr_mse)

print(f"The Root Mean Square Error is {svr_rmse:.2f}")
data = test_ab.iloc[:5]

labels = test_labels.iloc[:5]



print("Predictions:" ,svr_ab.predict(data))

print("Labels:", list(labels))