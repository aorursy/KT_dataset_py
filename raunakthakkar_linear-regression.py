# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
trainingData=pd.read_csv(r'/kaggle/input/random-linear-regression/train.csv')

testData=pd.read_csv(r'/kaggle/input/random-linear-regression/test.csv')

trainingData.head()
# checking the null count

trainingData.isnull().sum(axis=0)
trainingData.dropna(inplace=True)

trainingData.isnull().sum(axis=0)
plt.scatter(trainingData['x'],trainingData['y'])

plt.show()
import statsmodels.api as sm
# creating a linear regression model now because there is no data to clean and the required visualisation is done

# and also not need to scale the variable because there is only one variable

# creating a model using statsModels package (because it adds constant and calculates the value of y when x=0)

#using statsmodels ols method that is odinary least square method
trainingData2=sm.add_constant(trainingData)

trainingData2=trainingData2.drop('y',axis=1)

trainingData2.head()
dependentVar=trainingData['y']

dependentVar.head()
trainingModel=sm.OLS(dependentVar,trainingData2).fit()

trainingModel.summary()
testData2=sm.add_constant(testData)

testData2=testData.drop('y',axis=1)

testData2.head()

traningDataPred=trainingModel.predict(trainingData2)

traningDataPred.head()
finalPred=trainingModel.predict(testData)
finalPred.head()
fig, ax = plt.subplots()

ax.plot(trainingData2['x'], dependentVar, 'o', label="Data")

ax.plot(testData['x'], testData['y'], 'b-', label="True")

ax.plot(np.hstack((trainingData2['x'], testData['x'])), np.hstack((traningDataPred, finalPred)), 'r', label="OLS prediction")

ax.legend(loc="best");

plt.show()