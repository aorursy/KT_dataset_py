# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#Import dataset

dataset = pd.read_csv("../input/weather.csv")
#Encoding string values into Integer

from sklearn.preprocessing import LabelEncoder

labeEncoder_X=LabelEncoder()

for i in range(1,6):

    dataset.iloc[:, i]=labeEncoder_X.fit_transform(dataset.iloc[:, i])
#From dataset input values into X and Y

x=dataset.iloc[:, 1:5].values

y=dataset.iloc[:, 5:].values
#Create Linear Regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
#Fitting dataset

regressor.fit(x,y)
#Predict percent then is better go play

y_pred=regressor.predict(x)
#Visualising what days choice for Play

plt.scatter(dataset.iloc[:,0],y_pred,color='blue')

plt.title('Play or Not Play')

plt.xlabel('Days')

plt.ylabel('Predict percent')

plt.show()