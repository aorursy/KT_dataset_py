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

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



#Read Train dataset

data=pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')

x=data.iloc[:,5:-1]

y=data.iloc[:, -1]



#Create Linear regression model

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()



#Fitting train set on the model

regressor.fit(x,y)



#Predicting values with Test set

test=pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')

testOriginal=pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')

test=test.iloc[:,5:42]

y_pred=regressor.predict(test)















#Save submission

submission = pd.DataFrame({

    'Id' : testOriginal.Id,

    'Prediction' : y_pred

})

submission.to_csv('submission.csv', index=False)