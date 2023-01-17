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
#import the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#importing the dataset

my_data=pd.read_csv('../input/best-regression-model/Data.csv')

X=my_data.iloc[:,:-1].values

y=my_data.iloc[:,-1].values
print(X)
#splitting the datset

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Training the Multiple linear Regression

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(X_train,y_train)
#Predicting the result

y_pred = model.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)