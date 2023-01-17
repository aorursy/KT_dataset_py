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
#import libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import dataset:
dataset=pd.read_csv("../input/Salary_Data.csv")
dataset.head()
dataset.describe()
# make the dataset into independen form:
X=dataset.iloc[:,:-1].values
X
# make the datset into the dependent form:
y=dataset.iloc[:,1].values
y
# dataset visualization:
sns.pairplot(dataset)
sns.heatmap(dataset)
# spliting the dataset into train and test set:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
#fitting the model on linear_regression:
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#prediction of new result:
y_pred=regressor.predict(X_test)

y_pred
y_test
# visualixation of train dataset:
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Expriance')
plt.ylabel('salary')
plt.title('Expirence vs Salary')
plt.show()
# vizualixation of test dataset :
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Expirence Vs Salary')
plt.xlabel('Expirence')
plt.ylabel('Salary')
plt.show()

