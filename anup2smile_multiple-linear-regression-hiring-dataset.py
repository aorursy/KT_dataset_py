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

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/hiring.csv')



dataset.head()
dataset
dataset.experience

dataset.experience = dataset.experience.fillna('zero')



dataset.experience

from  word2number import w2n



dataset.experience = dataset.experience.apply(w2n.word_to_num)

dataset.experience
dataset.columns
import math

median_test = math.floor(dataset['test_score(out of 10)'].median())
median_test
dataset['test_score(out of 10)'] = dataset['test_score(out of 10)'].fillna(median_test)
dataset
X = dataset.iloc[: , :3].values



y = dataset.iloc[:, -1].values



X
y
from sklearn.linear_model import LinearRegression



regressor = LinearRegression()



regressor.fit(X, y)
regressor.coef_
regressor.intercept_
regressor.score(X, y)
regressor.predict([[5,6,7]])
regressor.predict([[2,10,10]])
regressor.predict([[2,9,6]])
X,y