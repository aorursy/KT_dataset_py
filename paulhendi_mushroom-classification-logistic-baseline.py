# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_df = pd.read_csv("../input/mushrooms.csv")
data_df.sample(5)
data_df.info() # All col types are object
# Checking for Nan
print("Nb of Nan in the whole dataset : %d" %np.sum(data_df.isnull().sum()))

# Checking for constant in feature cols
print("_"*50)
for i in data_df.columns : 
    print(data_df[i].value_counts())
    print("_"*50)
# Removing constant feature veil-type
data_df.drop(["veil-type"], axis = 1, inplace = True) 
# Now we need to encode all the features to integers
for i in data_df.columns : 
    data_df[i] = data_df[i].factorize()[0].astype(int)

data_df.sample(5)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X = data_df.drop("class", axis = 1)
y = data_df["class"]

clf = LogisticRegression(C = 100)
print(np.mean(cross_val_score(clf, X, y, cv = 5)))
# Checking which weight are more important for the logReg
clf.fit(X,y)

coefs = np.array(np.sort(clf.coef_))[0][::-1]
cols = X.columns[np.array(np.argsort(clf.coef_))[0][::-1]]
print(cols)
print(coefs)
import matplotlib.pyplot as plt
%matplotlib inline


fig,ax=plt.subplots(figsize = (40,40))
ax.set_xticks(np.arange(len(coefs)))
ax.set_xticklabels(cols)
plt.bar(np.arange(len(coefs)), coefs)