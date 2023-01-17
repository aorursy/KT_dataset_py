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
dataset = pd.read_csv("../input/train.csv")

print(dataset.head(20))



import matplotlib.pyplot as plt   #Data visualisation libraries 

import seaborn as sns
dataset.head()
dataset.info()
dataset.describe()
dataset.columns
import seaborn as sns

sns.distplot(dataset['SalePrice'])
dataset.corr()
corr = dataset.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
X = dataset[['LotArea']]

y = dataset['SalePrice']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
import matplotlib.pyplot as plt

plt.scatter(y_test,predictions)