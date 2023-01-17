# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr



%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



all_data
matplotlib.rcParams['figure.figsize'] = (6.0, 3.0)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices.hist()
#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



print(numeric_feats)



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

print(skewed_feats)

print("----------------------")



skewed_feats = skewed_feats[skewed_feats > 0.75]

print(skewed_feats)

print("----------------------")



skewed_feats = skewed_feats.index

print(skewed_feats)

print("----------------------")



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])



print(all_data[skewed_feats])

print("----------------------")

all_data = pd.get_dummies(all_data)



all_data
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())



all_data
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice



print(X_train)

print("----------------------")

print(X_test)

print("----------------------")

y