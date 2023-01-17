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

Marks = pd.read_csv("../input/marks1/Marks.csv")

import numpy as np

from sklearn import linear_model

from sklearn.model_selection import train_test_split



X=Marks['internal_marks']

Y=Marks['external_marks']

cov_XY=np.cov(X,Y)[0][1]

var_X=np.var(X)

mean_X=np.mean(X)

mean_Y=np.mean(Y)



b=cov_XY/var_X

print("Value of b:",b)

a=mean_Y-b*mean_X

print("Value of a:",a)

print("Simple linear regression modei is: y=",a,"+",b,"X")

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import linear_model



data = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")



data = data.dropna(axis=0, how='any') #Remove all rows where value of any column is ‘NaN’

predictors = data.iloc[:,1:7] # Seggretating the predictor variables ...

target = data.iloc[:,0] # Seggretating the target / class variable ...

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.3, random_state = 123)

lm = linear_model.LinearRegression()



# First train model / classifier with the input dataset (training data part of it)

model = lm.fit(predictors_train, target_train)



# Make prediction using the trained model

prediction = model.predict(predictors_test)

mse = mean_squared_error(target_test, prediction)

r2s = r2_score(target_test, prediction)



print("Mean squared error: ", mse)

print("R2 score: ", r2s)

print(lm.intercept_)
