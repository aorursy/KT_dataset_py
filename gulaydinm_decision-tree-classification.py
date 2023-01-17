# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data (1).csv")
data.info()
data.head(10)
data.drop(["id" , "Unnamed: 32"] ,axis = 1 , inplace = True)
data.tail()
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
data.info()
y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis = 1)
#normalization

x = ((x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.15 , random_state = 42)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=1)

dt.fit(x_train,y_train)



print("score is {}".format(dt.score(x_test,y_test)))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100 , random_state = 1)

rf.fit(x_train,y_train)



print("score via random forest is {}".format(rf.score(x_test,y_test)))