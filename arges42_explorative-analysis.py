import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



% matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
pd.options.display.max_columns = 999

train.head()
print("Train shape:{}".format(train.shape))

print("Test shape:{}".format(test.shape))
train.dtypes.value_counts()
missing = pd.DataFrame(columns=["Missing", "Percent"])

missing.loc[:,"Missing"] = train.isnull().sum()[train.isnull().sum()>0]

missing.loc[:,"Percent"] =  missing.loc[:,"Missing"]/train.shape[0]

missing.sort_values("Percent", ascending=False)
sns.distplot(train.SalePrice)
sns.distplot(np.log(train.SalePrice))
correlation = train.corr()["SalePrice"].sort_values(ascending=False)

correlation
sns.pairplot(train[correlation[correlation>0.62].keys()])