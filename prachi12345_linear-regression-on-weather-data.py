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
import seaborn as sns
from sklearn import linear_model
df=pd.read_csv("../input/ml4.csv")
print(df)
df.plot()
linear = linear_model.LinearRegression()
trainX = np.asarray(df.Rainfall[0:15]).reshape(-1, 1)
trainY = np.asarray(df.humidity[0:15]).reshape(-1, 1)
testX = np.asarray(df.Rainfall[15:21]).reshape(-1, 1)
testY = np.asarray(df.humidity[15:21]).reshape(-1, 1)
linear.fit(trainX, trainY)
linear.score(trainX, trainY)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print('R² Value: \n', linear.score(trainX, trainY))
predicted = linear.predict(testX)
predicted
print(testX)

