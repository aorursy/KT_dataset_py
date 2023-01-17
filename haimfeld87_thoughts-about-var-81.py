# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
sns.distplot(train["var_81"], bins=100)
train["age"] = train["var_81"]*(np.log(2*train["var_81"]))

print("min value " + str(min(train["age"])))

print("max value " + str(max(train["age"])))
sns.distplot(train["age"], bins=100)
test["age"] = test["var_81"]*(np.log(2*test["var_81"]))

print("min value " + str(min(test["age"])))

print("max value " + str(max(test["age"])))
sns.distplot(test["age"], bins=100)