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
import numpy as np # linear algebra

import pandas as pd
import os

print(os.listdir("../input"))
df = pd.read_csv("../input/insurance.csv")
df.shape
from sklearn.preprocessing import LabelEncoder

labelencode = LabelEncoder()

df.iloc[:,4]= labelencode.fit_transform(df.iloc[:,4])
from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score
#These three features have relationship with expenses.

x = df[['age','bmi','smoker']]

y = df['expenses']

#train_test_split() to split the dataset into train and test set at random.

#test size data set should be 30% data

X_train,X_test,Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=42)
df.head()
print ('Train set:', X_train.shape,  Y_train.shape)

print ('Test set:', X_test.shape,  Y_test.shape)
k = 5

#Train Model and Predict  

neigh = KNeighborsRegressor(n_neighbors = k).fit(X_train,Y_train)

neigh
yhat = neigh.predict(X_test)

ytrain = neigh.predict(X_train)
from sklearn.metrics import mean_absolute_error,mean_squared_error
print("MSE:",np.sqrt(mean_squared_error(Y_train, ytrain)))

print("MSE:",np.sqrt(mean_squared_error(Y_test, yhat)))

#print("MSE only for Smoker:", np.sqrt(mean_squared_error(Y_train,smoker_model.predict(X_train[['smoker']]))))