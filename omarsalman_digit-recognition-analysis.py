# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Reading in the training and testing files

# using pandas into a dataframe

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



# Separating the features and labels for the

# training data

x_train = train_data.drop("label", axis=1)

y_train = train_data["label"]



# Separating the features and labels for the

# testing data

x_test = test_data



print("Data loaded successfully")
# Using SVM for classification

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 100)

clf.fit(x_train, y_train)

results = clf.predict(x_test)

output = pd.DataFrame(results, index=range(1, 28001), columns=["Label"])

output.index.names = ["ImageId"]

print(output)

output.to_csv("random_forest.csv")