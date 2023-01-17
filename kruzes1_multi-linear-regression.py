# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, r2_score

from  ipywidgets import interact
df = pd.read_csv("../input/insurance-1/insurance 1.csv")

df.head()
x = df.iloc[:,0:2].values

y = df.iloc[:,-1].values
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3, random_state = 16)
model = LinearRegression()

model.fit(xtrain,ytrain)
m = model.coef_

c = model.intercept_
x = [12, 37]

y_predict = sum(m*x) + c

y_predict
x = [12, 37]

y_predict = model.predict([x])

y_predict
ytest_predict = model.predict(xtest)
def InsurancePricePredict(Age, bmi):

    y_predict = model.predict([[Age, bmi]])

    print("Insurance Price is:", y_predict[0])
InsurancePricePredict(33,3)
age_min = df.iloc[:, 0].min()

age_max = df.iloc[:, 0].max()

bmi_min = df.iloc[:, 1].min()

bmi_max = df.iloc[:, 1].max()
interact(InsurancePricePredict, Age = (age_min, age_max), bmi = (bmi_min, bmi_max) )