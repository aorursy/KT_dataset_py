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
import numpy as np    #numpy is a library for making computations
import matplotlib.pyplot as plt    #it is a 2D plotting library
import pandas as pd    # pandas is mainly used for data analysis
import seaborn as sns    # data visualization library
%matplotlib inline 
#magic function to embed all the graphs in the python notebook
#Import the salary dataset
#Reading the csv file using the read_csv function of the pandas module
df=pd.read_csv("../input/Salary_Data.csv")
#The read_csv function converts the data into dataframe
#Look how the data looks like
#Lets print the first 5 rows of the dataframe
df.head()
X=df.iloc[:,:-1].values
#Storing the column 1 in X and column 2 in y
y=df.iloc[:,:1].values
sns.distplot(df['YearsExperience'],kde=False,bins=10)
#This plot is used to represent univariate distribution of observations
#Show the counts of observations in each categorical bin using bars
sns.countplot(y='YearsExperience',data=df)
#Plotting a barplot
sns.barplot(x='YearsExperience',y='Salary',data=df)
#Representing the correlation among the columns using a heatmap
sns.heatmap(df.corr())
sns.distplot(df.Salary)
from sklearn.model_selection import train_test_split
#splitting the data using this module and setting the test size as 1/3 . Rest 2/3 is used for training the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#importing the linear regression model
from sklearn.linear_model import LinearRegression
#creating the model
lr=LinearRegression()

lr.fit(X_train,y_train)
#fitting the training data

X_train.shape 
#Counting the number of observations in the training data
y_train.shape

y_pred=lr.predict(X_test)
y_pred
#Predicted data
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,lr.predict(X_train),color='red')
plt.title('Salary vs Years of Experience (Training Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary of an employee')
plt.show()
plt.scatter(X_test,y_test,color='blue')
plt.plot(X_test,lr.predict(X_test),color='red')
plt.title('Salary vs Years of Experience (Test Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary of an employee')
plt.show()
from sklearn import metrics
print('Mean Absolute Error of the Model:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error of the Model: ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error of the Model: ',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
#This shows that our model is completely accurate
#R value lies between 0 to 1. Value of 1 represents it is completely accurate
