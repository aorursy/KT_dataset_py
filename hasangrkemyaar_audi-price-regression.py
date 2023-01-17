import pandas as pd

import numpy as np

import seaborn as sns

import sklearn as sk

from sklearn import metrics

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv")

df.head(11)
#checking the null values

print(df.isnull().sum())
#To see the relation of these features

sns.pairplot(df)

plt.show()
f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, linewidth=.25, fmt='.1f', ax=ax);
# Count plot on model

plt.subplots(figsize=(10,10))

ax = sns.countplot(df.model, label = "Count")
plt.subplots(figsize=(10,10))

price_by_model = df.groupby("model")['price'].mean().reset_index()

plt.title("Average Price of vechicle")

sns.set()

sns.barplot(x="model", y ="price", data = price_by_model)

plt.show()

#creating mileage and price arrays to make a linear regression of them

x = np.array(df.loc[:,"engineSize"]).reshape(-1,1)

y = np.array(df.loc[:,"price"]).reshape(-1,1)
#scatter

plt.figure(figsize=[10,10])

plt.scatter(x,y)

plt.xlabel("engineSize")

plt.ylabel("price")

plt.show()
#linear regression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

# Predict space

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

#fit

reg.fit(x,y)

#predict

prediction = reg.predict(predict_space)

#finding the r^2 score

print("R^2 Score: ", reg.score(x,y))
plt.plot(predict_space, prediction, color='black', linewidth=5)

plt.scatter(x=x,y=y)

plt.xlabel('engineSize')

plt.ylabel('price')

plt.show()