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
import pandas as pd
dataset = pd.read_csv("../input/iris.csv")
# let’s see a sample of our data
dataset.head()
y = dataset["species"]

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = dataset[features]
from sklearn.model_selection import train_test_split

#splitting our dataset for training and validation
#random_state to shuffle our data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
train_y = pd.get_dummies(train_y)
val_y = pd.get_dummies(val_y)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=1)
random_forest.fit(train_X, train_y)
from sklearn.metrics import mean_absolute_error

preds = random_forest.predict(val_X)
mae = mean_absolute_error(val_y, preds)

print("Mean absolute error is: {:,.5f}".format(mae * 100))
print(random_forest.score(val_X, val_y) * 100)
