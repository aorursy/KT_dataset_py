import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print("Train data dimensions: ", train.shape)

print("Test data dimensions: ", test.shape)
# little preview of the data

train.head()
print(train.isnull().values.any())

print(test.isnull().values.any())
# putting continuous features together

contFeatureslist = []

for colName,x in train.iloc[1,:].iteritems():

    #print(x)

    if(not str(x).isalpha()):

        contFeatureslist.append(colName)
# remove id and loss for the purpose of graphing

contFeatureslist.remove("id")

contFeatureslist.remove("loss")
plt.figure(figsize=(13,9))

sns.boxplot(train[contFeatureslist])
plt.figure(figsize=(13,9))

sns.boxplot(np.log(train[contFeatureslist]))
plt.figure(figsize=(13,9))

sns.distplot(np.log1p(train["loss"]))
plt.figure(figsize=(13,9))

sns.distplot(train["loss"])

sns.boxplot(train["loss"])