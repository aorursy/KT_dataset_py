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
def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier





#In this project, we will be using a dataset containing census information from UCI’s Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).



#By using this census data with a random forest, we will try to predict whether or not a person makes more than $50,000.



#Investigate the data available to us.

#Since the first row of our file contains the names of the columns, we also want to add the argument header = 0.



income_data = pd.read_csv("../input/income.csv", header = 0, delimiter = ", ")





#Notice how the delimiter = ", " instead of "," because of the format of original csv

print(income_data.columns)

print(income_data.head())





#Transform strings into integers for sex column

income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)



#Since the majority of the data comes from "United-States", it might make sense to make a column where every row that contains "United-States" becomes a 0 and any other country becomes a 1. Use the syntax from creating the "sex-int" column to create a "country-int" column.



#When mapping Strings to numbers like this, it is important to make sure that continuous numbers make sense. For example, it wouldn’t make much sense to map "United-States" to 0, "Germany" to 1, and "Mexico" to 2. If we did this, we’re saying that Mexico is more similar to Germany than it is to the United States.



#However, if you had values in a column like "low", "medium", and "high" mapping those values to 0, 1, and 2 would make sense because their representation as Strings is also continuous.





#Transform strings into integers for native-country column

print(income_data["native-country"].value_counts())

income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)







#Print income_data.iloc[0] to see the first row in its entirety. 

#Did this person make more than $50,000? 

#What is the name of the column that contains that information?



print(income_data.iloc[0])



#Put the data in a format that our Random Forest can work with. 



labels = income_data[["income"]]



features = ["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]

data = income_data[features]



#Split our data and labels into a training set and a test set. 

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)



#Hooray!!! Build and test our random forest. 

forest = RandomForestClassifier(random_state = 1)



forest.fit(train_data, train_labels)



#This will show you a list of numbers where each number corresponds to the relevance of a column from the training data. Which features tend to be more relevant?

print(forest.feature_importances_)



from matplotlib import pyplot as plt

plt.figure(figsize=(15,12))

plt.bar(features, forest.feature_importances_)

plt.xlabel("feature")

plt.ylabel("importance")

plt.title("forest feature importances")

plt.show()



print(forest.score(test_data, test_labels))



#print(forest.predict(test_data))







#Next step: Create a tree.DecisionTreeClassifier, train it, test is using the same data, and compare the results to the random forest. When does the random forest do better than the single tree? When does a single tree do just as well as the forest?



#Next step: Use some of the other columns that use continuous variables, or transform columns that use strings!




