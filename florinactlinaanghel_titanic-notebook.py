# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex3 import *

print("Setup completed :)")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read the data from the test.csv

data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

print(data.head(20))

print(data.Survived)

print(test_data.head(20))
# select the most important columns

data_features = ["Pclass", "Age", "SibSp", "Parch"]



# select the desired info

X = data[data_features]

y = data.Survived

z = data["Survived"]

print(X.head())
# creating, fitting the model and predict the result

model = DecisionTreeRegressor(random_state = 1)

X = X.apply (pd.to_numeric, errors='coerce')

X = X.dropna(axis = 1)

model.fit(X, y)



predictions = model.predict(X)



match = 0

count = 0

for prediction, value in zip(predictions, z):

    print(int(round(prediction)), value)

    if int(round(prediction)) == value:

        match += 1

    count += 1

print("Acuracy: ", match/count *100 )