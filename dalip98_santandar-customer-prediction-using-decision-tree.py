# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv("../input/train.csv")

print(train_dataset.head(5))
#loading the training set

x = train_dataset.iloc[: , 2:].values

y = train_dataset.iloc[: , 1].values
train_dataset['target'].value_counts().head(10).plot.bar()
#loading the test set

test_dataset = pd.read_csv('../input/test.csv')

x_test = test_dataset.iloc[: , 1:].values
#fitting on our model

from sklearn.tree import DecisionTreeRegressor

classifier = DecisionTreeRegressor(random_state = 0)

classifier.fit(x , y)
#Predicting

y_pred = classifier.predict(x_test)



print(y_pred)