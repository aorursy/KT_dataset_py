# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
train.head()
test.describe()
test["Fare"].mean()
sex_pivot = train.pivot_table(index = "Sex", values = "Survived")
sex_pivot.plot.bar()
plt.show()
survived = train[train["Survived"]==1]
died = train[train["Survived"]==0]
survived["Age"].plot.hist(alpha = 0.5,color = 'green',bins = 50)
died["Age"].plot.hist(alpha = 0.5,color = 'black',bins = 50)
plt.legend(['Survived','Died'])
plt.show()