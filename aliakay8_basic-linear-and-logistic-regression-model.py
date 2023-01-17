# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Data Generation
x = np.arange(10,100,step=5)
x_test=np.arange(100)
y = 2*x+5+20*np.random.randn(18)
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
##İmport Linear Regression Libraries
from sklearn.linear_model import LinearRegression
model=LinearRegression()
x=x.reshape(-1,1)
model.fit(x,y)


##Apply ML Algorithm
##Learn Parameters
##Predict
model.fit(x,y)

#Test Split
x_test=np.arange(100)
x_test=x_test.reshape(-1,1)

y_pred = model.predict(x_test)
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_test,y_pred,'r')

#Evaluation
model.score(x,y)
## Logistic Regression
from sklearn.linear_model import LogisticRegression

##Collect Data
x0 = np.random.randn(100)+2
x1 = np.random.randn(100)+2

x0_=np.random.randn(100)+3 ##if you increase the number you can see that the datas has grouped, if you decrease the number the data will so complicated.
x1_=np.random.randn(100)+3

xx0=np.concatenate((x0,x0_))
xx1=np.concatenate((x1,x1_))
y = np.concatenate((np.zeros(100),np.ones(100)))

d= {"x0":xx0,"x1":xx1,"y":y}
data=pd.DataFrame(d)
data.head(10)

c=['b','r']
mycolors = [c[0] if i==0 else c[1] for i in y]
data.plot.scatter('x0','x1',c=mycolors)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
len(X_train), len(y_test)
# Apply Machine learning algorithm
model = LogisticRegression()
# learn parameters with fit
model.fit(X_train, y_train)
# prediction
y_pred = model.predict(X_test)
#Evaluation
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

#Confusion_Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
