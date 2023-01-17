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
#importing necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
#Reading the csv file

datam=pd.read_csv(os.path.join(dirname, filename))

datam
datam.columns
# These are how I assume the columns means

# sl_no = Serial Number

# gender = Male or Female

# ssc_p = senior sceondary percentage (10)

# ssc_b = senior secondary board

# hsc_p = higher secondary percetage (12)

# hsc_b = higher secondary board (12)

# hsc_s = higher secondary stream (12)

# degree_p = degree percentage (UG)

# degree_t = degree type (UG)

# workex = work experience

# etest_p = its a test percentage

# specialisation = the specialisation you are doing in MBA

# mba_p = MBA Percentage

# status = are you placed or not

# salary = package 
print("dataset size:{0}\ndataset dimension: {1}\ndataset Shape{2}".format(datam.size, datam.ndim, datam.shape))
for i in datam:

    print("{0}={1}".format(i,datam[i].unique()))
datam["salary"]=datam['salary'].fillna((-1)) #column.fillna((value)) to replace nan with value in that column
datam["salary"].unique()
plt.hist(datam["sl_no"], edgecolor="#7FFFD4")

plt.title("Serial Number Histogram")
sns.countplot(datam['gender'], color="#FF4040", edgecolor="#00008B")

plt.title("Gender Countplot")
plt.hist(datam["ssc_p"], color="#FF4040", edgecolor="#00008B")

plt.title("Senior Seconday Percentage Histogram")
sns.countplot(datam["ssc_b"], color="#FF4040", edgecolor="#00008B")

plt.title("Senior Seconday Board Countplot")
plt.hist(datam["hsc_p"], color="#FF4040", edgecolor="#00008B")

plt.title("Higher Seconday Percentage Histogram")
sns.countplot(datam["hsc_b"], color="#FF4040", edgecolor="#00008B")

plt.title("Higher Seconday Board Countplot")
sns.countplot(datam["hsc_s"], color="#FF4040", edgecolor="#00008B")

plt.title("Higher Seconday Stream Countplot")
plt.hist(datam["degree_p"], color="#FF4040", edgecolor="#00008B")

plt.title("Degree percentage Histogram")
sns.countplot(datam["degree_t"], color="#FF4040", edgecolor="#00008B")

plt.title("Degree Type countplot")
#lets make a copy of dataset so that we have an original intact

datam1=datam.copy(deep=True)
"""  #Task 1:



1) Develop an estimated multiple linear regression equation with mbap as response variable 

and sscp & hscp as the two predictor variables. 

Interpret the regression coefficients and check whether they are significant based on the summary output """
""" #Task 2

2) Estimate a multiple regression equation for each of the below scenarios and based on the model’s R-square comment which model is better. 

(i) Use mbap as outcome variable and sscp & degreep as the two predictor variables.

(ii) Use mbap as outcome variable and hscp & degreep as the two predictor variables. 

"""
""" #Task 3

3) Show the functional form of a multiple regression model. 

Build a regression model with mbap as dependent variable and sscp, hscp and degree_p as three independent variables. 

Divide the dataset in the ratio of 80:20 for train and test set (set seed as 1001) and use the train set to build the model. 

Show the model summary and interpret the p-values of the regression coefficients. 

Remove any insignificant variables and rebuild the model. 

Use this model for prediction on the test set and 

show the first few observations’ actual value of the test set in comparison to the predicted value."""
datam1.plot('ssc_p','mba_p',color="#CD1076",style="*")

plt.title("Relation b/w Senior Secondary Percentage & MBA Percentage")

plt.xlabel("Senior Secondary Percentage")

plt.ylabel("MBA Percentage")

plt.show()
datam1.plot('hsc_p','mba_p',color="#CD1076",style="+")

plt.title("Relation b/w Higher Secondary Percentage & MBA Percentage")

plt.xlabel("Higher Secondary Percentage")

plt.ylabel("MBA Percentage")

plt.show()
from sklearn.linear_model import LinearRegression

from scipy import stats

from statsmodels.formula.api import ols
#First we will use LinearRegression

x=datam1[['ssc_p','hsc_p']]

y=datam1['mba_p']
reg=LinearRegression()

result=reg.fit(x,y)
result.coef_
result.intercept_
#Now we will use statsmodel to do the same

res=ols(formula="mba_p ~ ssc_p+hsc_p", data=datam1).fit()

res.summary()
#ploting the residual plot

f, axes = plt.subplots(2, figsize=(7, 7), sharex=True)

sns.residplot(datam1['ssc_p'],y,lowess=True, ax=axes[0])

sns.residplot(datam1['hsc_p'],y,lowess=True,ax=axes[1])
sns.regplot(datam1['degree_p'], datam1['mba_p'])

plt.title("Relation b/w Degree Percentage & MBA Percentage")

plt.xlabel("Degree Percentage")

plt.ylabel("MBA Percentage")

plt.show()
res1=ols(formula="mba_p ~ ssc_p+degree_p", data=datam1).fit()

res1.summary()
res2=ols(formula="mba_p ~ hsc_p+degree_p", data=datam1).fit()

res2.summary()
print("R2 value of res1(ssc_p & degree_P)= {0}".format(res1.rsquared))

print("R2 value of res2(hsc_p & degree_P)= {0}".format(res2.rsquared))
# importing necessary libraries

from sklearn.model_selection import train_test_split

from sklearn import metrics
#making independet & dependent variables

X=datam1[['ssc_p','hsc_p','degree_p']]

Y=datam1['mba_p']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1001)
Regress=LinearRegression()

Regress.fit(X_train, Y_train)
Regress.coef_
Regress.intercept_
ypredict=Regress.predict(X_test)
# now lets compare

datapredict = pd.DataFrame({'Actual': Y_test, 'Predicted': ypredict})

datapredict
#As we have so many values, we will be showing Actual VS Predicted of 1st 10

datapredict1=datapredict.head(10)

datapredict1
datapredict1.plot(kind='barh', figsize=(15,7))