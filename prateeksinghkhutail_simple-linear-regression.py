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
import matplotlib.pyplot as plt#to visualize the data

# Importing the dataset

dataset=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

dataset
#now first we will try simple linear regression to find the relationship between mbap (outcome variable) and degreep (predictor variable).

X=dataset.iloc[:,7].values.reshape(-1,1)

Y=dataset.iloc[:,12].values.reshape(-1,1)

# we have to use array.reshape(-1,1) whenever we our array have single feature otherwise it will throw an error

#what it does is  it let numpy automatically reshape array.
X
Y
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
#Fitting simple learn regression to the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()



regressor.fit(X_train,Y_train)



#predicting the test set results

y_pred=regressor.predict(X_test)
df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_pred.flatten()})

df
#visualizing the training set results

plt.scatter(X_train,Y_train,color='red')

plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.title('mbap (outcome variable) and degreep (predictor variable)(Training set)')

plt.xlabel('degreep')

plt.ylabel('mbap')

plt.show()
#visualizing the test set results

plt.scatter(X_test,Y_test,color='red')

plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.title('mbap (outcome variable) and degreep (predictor variable)(Test set)')

plt.xlabel('degreep')

plt.ylabel('mbap')

plt.show()