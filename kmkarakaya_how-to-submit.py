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
train= pd.read_csv("../input/first-challenge-find-the-output-number/train.csv")
test = pd.read_csv("../input/first-challenge-find-the-output-number/test.csv")
train.head()
test.tail()
train.info()
## X usually means our input variables (or independent variables)
X = train.iloc[:,:-2]
## Y usually means our output/dependent variable
y_A= train.iloc[:,6:7]
y_B= train.iloc[:,7:]
print("X", X.head())
print("y_A",y_A.head())
print("y_B",y_B.head())



import statsmodels.api as sm
#X = sm.add_constant(X)
model_A = sm.OLS(y_A, X).fit()
print("modelA",model_A.summary())

model_B = sm.OLS(y_B, X).fit()
print("modelB",model_B.summary())
X= test.iloc[:,1:]
predictionsForA = model_A.predict(X)
predictionsForB = model_B.predict(X)





predictionsForA.head()
my_submission = pd.DataFrame({ 'ID': test.ID,'A': predictionsForA, 'B': predictionsForB})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
my_submission.head()
from IPython.display import Image
Image("../input/submission/submission.png")