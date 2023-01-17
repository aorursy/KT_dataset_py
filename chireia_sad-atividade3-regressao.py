import os

import requests

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
print(os.listdir("../input"))
df = pd.read_csv("../input/vivareal-listings-cpa.csv")

df.head()
plt.figure(figsize=(5,5))

sns.heatmap(df.corr() ,annot=True ,linewidths=0 ,cmap='coolwarm', cbar=False)

plt.show()
features = ['price', 'usableAreas']

PricePerAreaFrame = df[features]

PricePerAreaFrame.head()
sns.pairplot(data=PricePerAreaFrame, kind="reg")
x = df[features[1:]]

y = df.price.values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

print(len(x_train),len(x_test))
scaler = preprocessing.MaxAbsScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train.values.reshape(-1, 1)),columns=features[1:])

x_test = pd.DataFrame(scaler.transform(x_test.values.reshape(-1, 1)),columns=features[1:])
lr = LinearRegression()

lr.fit(x_train,y_train)
parametros = [[250]]



parametros = pd.DataFrame(scaler.transform(parametros),columns=features[1:])

print("Valor do imóvel: $ %.2f" % lr.predict(parametros.values))