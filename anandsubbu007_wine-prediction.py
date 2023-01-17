%%time

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/predict-red-wine-quality/train.csv")

test = pd.read_csv("../input/predict-red-wine-quality/test.csv")

train.head(3)
train.info()

# all are numeric values
train.describe()
#Drop useles data

train1 = train.drop(["id"],axis=1)

test1 = test.drop(["id"],axis=1)

train1_col = list(train1.columns)
#sns.boxplot(column=test1_col,data = test1)



for i in train1.columns:

    train1.boxplot(column=i)

    plt.show()
q1 = train1.quantile(0.25)

q3 = train1.quantile(0.75)

iqr = q3-q1

l = q1-1.5*iqr

h = q3+1.5*iqr

#train2 = train1.loc[(train1 > l) & (train1 < h).any(axis=1)]

#train2



train2 = train1[((train1 >= (q1 - 1.5 * iqr))& (train1 <= (q3 + 1.5 * iqr))).all(axis=1)]

train2.describe()
for i in train2.columns:

    train2.boxplot(column=i)

    plt.show()
train2.info()
sns.pairplot(train2, diag_kind="reg")
train1_col
sns.pairplot(train2, x_vars=['fixed.acidity','volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 

                             'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 

                             'alcohol'], y_vars='quality', markers="+",size=5)
plt.subplots(figsize=(10,10))

sns.heatmap(train2.corr(), linewidths=1, annot=True)
train2.corr()["quality"].sort_values()
train2 = train2.drop(["residual.sugar"],axis=1)

train2 = train2.drop(["free.sulfur.dioxide"],axis=1)

train2 = train2.drop(["pH"],axis=1)
from sklearn.model_selection import train_test_split

x = train2.drop(['quality'],axis=1)

y = test.drop(["id"],axis=1)

#y_test = test["quality"]

x_train = train2.drop(['quality'],axis=1)

y_train = train2["quality"]

x_test = test.drop(["id"],axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = x_train.columns

vif['VIF'] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
from sklearn.linear_model import LinearRegression

model = LinearRegression()

#model.fit(x_train,y_train)
from sklearn.feature_selection import RFE

rfe = RFE(model, 6)

rfe.fit(x_train, y_train)
x_col = x_train.columns[rfe.support_]

x_col
model.fit(x_train[x_col],y_train)

#model.fit(x_train,y_train)
rst = model.predict(x_test[x_col])

#model.predict(x_test)
rst = rst.round(0)
ids = test['id']
file = open("result.csv", "w")

file.write("id,quality\n")

    

for id_, pred in zip(ids, rst):

    file.write("{},{}\n".format(id_, pred))

file.close()