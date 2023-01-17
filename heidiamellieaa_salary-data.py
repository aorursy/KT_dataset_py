# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Salary_Data = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")
display(Salary_Data.head(30))
pd.options.display.max_columns = None

display(Salary_Data.head(30))
x=pd.DataFrame(Salary_Data.YearsExperience)

y=pd.DataFrame(Salary_Data.Salary)
x.describe()
#fitting model, it produces x and y parameters

rl=linear_model.LinearRegression()

rl.fit(x, y)

#model to do a prediction

prediction = rl.predict(x)
prediction[1]
display(Salary_Data.Salary[1])
#knowing intercept value

rl.intercept_
#knowing slope value

rl.coef_