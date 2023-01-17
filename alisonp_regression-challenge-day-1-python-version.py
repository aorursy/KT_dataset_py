#Import the necessary libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

from scipy import stats

import seaborn as sns
# Read the data

recpies = pd.read_csv("../input/epi_r.csv")
# Drop any rows with a null value

recpies.dropna(inplace = True)
# Remove any rows that have alories greater than 10,000

recpies = recpies[recpies["calories"] <= 10000]

#recpies["calories"].max()
# check the ratings are all numeric

print("Is this variable numeric?")

np.issubdtype(recpies['rating'].dtype, np.number)
# check the ratings are integers(floats)

print("Is this variable an integer?")

np.issubdtype(recpies['rating'].dtype, np.integer)
# plot calories by whether or not it's a dessert

plt.figure(figsize=(10,10))

plt.scatter(recpies['calories'], recpies['dessert'])
X = recpies[["calories"]]

y = recpies["dessert"]

clf = linear_model.LogisticRegression()

clf.fit(X, y)
#create evenly spaced numbers

X_test = np.linspace(0, 10000)

def model(x):

    return 1 / (1 + np.exp(-x)) # calculate the expoential of each elemen
loss = model(X_test * clf.coef_ + clf.intercept_).ravel()

plt.figure(figsize=(10,10))

plt.scatter(recpies['calories'], recpies['dessert'])

plt.plot(X_test, loss, color='red', linewidth=3)
sns.lmplot(x="calories", y="dessert", data=recpies,

           logistic=True, size=10);
