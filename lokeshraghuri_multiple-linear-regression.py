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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv("../input/50-startups/50_Startups.csv")

dataset.head()

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

lable_encoder_X = LabelEncoder()

X[:,3] = lable_encoder_X.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features=[3])

X = onehotencoder.fit_transform(X).toarray()

X[0:5]

##float_formatter = lambda X :"%.2f" %X

   ## X_list = []

    ##for i in range(len(X)):

        ##X_level1 = []

       ## for j in range(6):

            ##X_level1.append(float_formatter(X[i][j]))

            ##X_list.append(X_level1)

##X_list[0:5]

X = X[:,1:]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

regressor.coef_

regressor.intercept_

y_pred = regressor.predict(X_test)

y_pred

y_test

##RMSC

np.sqrt(np.mean(y_test - y_pred)**2)

regressor.score(X_test,y_test)

regressor.score(X_train,y_train)


















