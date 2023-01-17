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
import pandas as pd

Salary_Data = pd.read_csv("../input/years-of-experience-and-salary-dataset/Salary_Data.csv")
Salary_Data
Salary_Data.describe()
Salary_Data.head(5)
Salary_Data.info()
import matplotlib.pyplot as plot
plot.scatter(Salary_Data['YearsExperience'],Salary_Data['Salary'])

plot.title('Salary vs Experience (Training set)')

plot.xlabel('Years of Experience')

plot.ylabel('Salary')

plot.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# Split the dataset into the training set and test set

# We're splitting the data in 1/3, so out of 30 rows, 20 rows will go into the training set,

# and 10 rows will go into the testing set.

xTrain, xTest, yTrain, yTest = train_test_split(Salary_Data.YearsExperience.values, Salary_Data.Salary.values, test_size = 1/3, random_state = 0)
xTrain
linearRegressor = LinearRegression()
linearRegressor.fit(xTrain.reshape(-1,1), yTrain)
yPrediction = linearRegressor.predict(xTest.reshape(-1,1))
plot.scatter(xTrain, yTrain, color = 'red')

plot.plot(xTrain, linearRegressor.predict(xTrain.reshape(-1,1)), color = 'blue')

plot.title('Salary vs Experience (Training set)')

plot.xlabel('Years of Experience')

plot.ylabel('Salary')

plot.show()
linearRegressor.get_params()

linearRegressor.coef_

linearRegressor.intercept_