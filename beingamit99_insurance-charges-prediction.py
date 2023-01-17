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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from ipywidgets import interact
df=pd.read_csv("../input/insurance.csv")
df.keys()
df.head()
x=df.iloc[:,[0,2,3]]
x
y=df.iloc[:,-1]
y
xtrain , xtest ,ytarin , ytest=train_test_split(x,y,test_size=0.3, random_state=11)
xtest
def Train_Test_Split(x,y,test_size = 0.5, random_state =None):

    n= len(y)

    if len(x)==len(y):

        if random_state:

            np.random.seed(random_state)

        shuffle_index =np.random.permutation(n)

        x=x[shuffle_index]

        y=y[shuffle_index]

        test_data =round(n * test_size)

        x_train , x_test = x[test_data :],x[: test_data]

        y_train , y_test = y[test_data :],y[: test_data]

        return x_train,x_test,y_train,ytest

    else:

        print("Data should be in same size")
xtrain , xtest , ytrain , ytest=train_test_split(x,y,test_size=0.9, random_state=11)
model =LinearRegression()
model.fit(xtrain,ytrain)
m = model.coef_

c = model.intercept_
x=[19,27.9,0]
y_predict=model.predict([x])
y_predict
ytest_predict = model.predict(xtest)
ytest_predict
ytest_predict
def insurance(Age ,Bmi , Children):

    y_predict = model.predict([[Age,Bmi,Children]])

    print("Insurance Charges", y_predict[0])
insurance(19,27.3,0)
age_min=df.iloc[:,0].min()
age_max=df.iloc[:,0].max()
age_min
age_max
bmi_min=df.iloc[:,2].min()
bmi_max=df.iloc[:,2].max()
child_min=df.iloc[:,3].min()

child_max=df.iloc[:,3].max()
interact(insurance , Age =(age_min ,age_max) ,

         Bmi=(bmi_min , bmi_max),

         Children = (child_min , child_max) )