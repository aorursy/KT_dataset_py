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
import matplotlib.pyplot as pl
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = pd.read_csv("../input/train.csv", sep=',') 
data1 = data.drop("id", 1)
data2 = data1.drop("diagnosis",1)
diag = data["diagnosis"]

Xtrain, Xtest, ytrain, ytest = train_test_split(data2, diag, test_size=0.2)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(Xtrain, ytrain)
predictions = lm.predict(Xtest)
writer = csv.writer(prediction, delimiter=',')


## The line / model
plt.scatter(ytest, predictions)



