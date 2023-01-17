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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,Lasso,Ridge

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.model_selection import train_test_split
#Importing dataset

sal=pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")
sal.head()
sal.shape
#Checking for presence of null values 

sal.isnull().sum()

#There are no null values
# To check whether any outliers are present in the dataset or not

sns.boxplot(x='YearsExperience',data=sal)

#We can see that no point is exceeding the (1.5*IQR) Range, So there are no outliers in the feature column
X=sal.drop(['Salary'],axis=1)

y=sal['Salary']
#For checking skewness in the data 

sns.distplot(sal['Salary'])

#There is no major skewness in the data set
#50% train and 50%test

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.5,random_state=7)
#Training in Linear Regression

clf=LinearRegression()

clf.fit(xtrain,ytrain)

pred=clf.predict(xtest)
#User defined function for checking MSE AND RMSE

def root(a,b,c):

    mse=np.square(np.subtract(a,b)).mean()

    rmse=np.sqrt(mse)

    print("Manual Calculation RMSE for test_size {} is {}:".format(c,rmse))
#RMSE FOR 50% SPLIT

root(ytest,pred,0.5)

print("Root Mean Squared Error for test_size=0.5 with metrics:",np.sqrt(mean_squared_error(ytest,pred)))
#R2SCORE FOR 50% SPLIT

print("R2 Score for 50% split is:",r2_score(ytest,pred))
#70% train and 30% test

xtrain1,xtest1,ytrain1,ytest1=train_test_split(X,y,test_size=0.3,random_state=7)
#Training 70% Split

clf.fit(xtrain1,ytrain1)

pred1=clf.predict(xtest1)
#RMSE AND R2 SCORE FOR 70% SPLIT

root(ytest1,pred1,0.3)

print("Root Mean Squared Error for test_size=0.3 with metrics:",np.sqrt(mean_squared_error(ytest1,pred1)))
print("R2 Score for 70% split is:",r2_score(ytest1,pred1))
#80% train and 20% test

xtrain2,xtest2,ytrain2,ytest2=train_test_split(X,y,test_size=0.2,random_state=3)
#Training 80% Split

clf.fit(xtrain2,ytrain2)

pred2=clf.predict(xtest2)
#RMSE AND R2 SCORE OF 80% Split 

root(ytest2,pred2,0.2)

print("Root Mean Squared Error for test_size=0.2 with metrics:",np.sqrt(mean_squared_error(ytest2,pred2)))
print("R2 Score for 80% split is:",r2_score(ytest2,pred2))
#Training in Ridge Regression for 50% Split

clf1=Ridge(alpha=0.05)

clf1.fit(xtrain,ytrain)

pred3=clf.predict(xtest)
#Calculating RMSE for Ridge Regression with 50% Split

root(ytest,pred3,0.5)

print("Root Mean Squared Error for test_size=0.5 with metrics:",np.sqrt(mean_squared_error(ytest,pred3)))
#Training with Ridge for 70% Split

clf1.fit(xtrain1,ytrain1)

pred4=clf1.predict(xtest1)
#Calculating RMSE for 70% Split Ridge Regression

root(ytest1,pred4,0.3)

print("Root Mean Squared Error for test_size=0.3 with metrics:",np.sqrt(mean_squared_error(ytest1,pred4)))
#Training with Ridge for 80% Split

clf1.fit(xtrain2,ytrain2)

pred5=clf.predict(xtest2)
#Calculating RMSE for 80% Split Ridge Regression

root(ytest2,pred5,0.2)

print("Root Mean Squared Error for test_size=0.2 with metrics:",np.sqrt(mean_squared_error(ytest2,pred5)))
#Since Lasso Makes the less important feature in a more complex dataset to 0, here in this dataset, just one feature is there so there is no point in using Lasso as it gives the same output as Linear Regression
#Tabulating all the RMSE for better understanding

df1=pd.DataFrame(columns=['Linear 50%','Ridge 50%','Linear 70%','Ridge 70%','Linear 80%','Ridge 80%'])

df1=df1.append({'Linear 50%':np.sqrt(mean_squared_error(ytest,pred)),'Ridge 50%':np.sqrt(mean_squared_error(ytest,pred3)),'Linear 70%':np.sqrt(mean_squared_error(ytest1,pred1)),'Ridge 70%':np.sqrt(mean_squared_error(ytest1,pred4)),'Linear 80%':np.sqrt(mean_squared_error(ytest2,pred2)),'Ridge 80%':np.sqrt(mean_squared_error(ytest2,pred5))},ignore_index=True)
df1