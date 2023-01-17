# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train_data = pd.read_csv ("../input/train.csv")

test_data = pd.read_csv ("../input/test.csv")

#print (train_data[["Fare", "Survived"]].groupby(["Fare"]).mean().sort_values (by = "Survived", ascending = False))

# Any results you write to the current directory are saved as output.



# Basic Assumption 1

# Assuming the first class people are more likely to survive



X = train_data [["Pclass", "Sex"]]

Y = train_data ["Survived"]

test = test_data [["Pclass", "Sex"]]



decision_tree = DecisionTreeClassifier()

decision_tree.fit (X, Y)



predict = decision_tree.predict (test);

dfPrediction = pd.DataFrame(data=predict,index = test_data.index.values,columns=["Survived"])



first_predicted_output = dfPrediction.to_csv("./first_output.csv", index = False)
