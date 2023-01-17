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

import pandas as p # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.formula.api as smf
import seaborn as sns
from matplotlib.pyplot import figure, show
from sklearn import datasets, linear_model
mydata=p.read_csv('../input/StudentsPerformance.csv') 
mydata
#head
mydata.head(10)
#change column names at a time
mydata.columns=['gender','race','par_ethnicity','lunch','pre_course','math_score','reading_score','writing_score']
mydata.head(10)
#size of a dataframe
print(mydata.shape)
#summary of dataframe
mydata.describe()
#to check missing value
mydata.isna().any()
mydata.isnull().sum()
#histogram- math_score by gender  
width=20
height=30
figure(figsize=(width,height))
sns.countplot(x="math_score",hue="gender",data=mydata)
show()

#histogram- writing_score by gender  
width=20
height=30
figure(figsize=(width,height))
sns.countplot(x="writing_score",hue="gender",data=mydata)
show()
#histogram- reading_score by gender  
width=20
height=30
figure(figsize=(width,height))
sns.countplot(x="reading_score",hue="gender",data=mydata)
show()

#boxplot for score and test prep by gender 
width=12
height=13
figure(figsize=(width,height))
sns.boxplot(data=mydata,x='gender',y='math_score',hue='pre_course')
show()

width=12
height=13
figure(figsize=(width,height))
sns.boxplot(data=mydata,x='gender',y='writing_score',hue='pre_course')
show()


width=12
height=13
figure(figsize=(width,height))
sns.boxplot(data=mydata,x='gender',y='reading_score',hue='pre_course')
show()

#Summary for boxplot visualizations:
#students who completed the pre_course had better scores in all three tests. 
# male students have received better scores in Math while female students in reading and writing.
#there is a presence of outliers in all three tests.

#model-Linear regression to predict math_score
lm1 = smf.ols(formula='math_score~writing_score+gender+race+lunch+pre_course+par_ethnicity', data=mydata).fit()
lm1.params

print(lm1.summary())
#Plotting the Least Squares Line
sns.pairplot(mydata, x_vars=['writing_score','reading_score'], y_vars='math_score', size=7, aspect=.7, kind='reg')

# confidence interval 
lm1.conf_int()
# print the R-squared value for the model
lm1.rsquared# print the p-values for the model coefficients
lm1.pvalues
# print the p-values for the model coefficients
lm1.pvalues

#Interpreting p-values
#If the 95% confidence interval does not include zero
#p-value will be less than 0.05
#Reject the null
#There is a relationship
#If the 95% confidence interval includes zero
#p-value for that coefficient will be greater than 0.05
#Fail to reject the null
#There is no relationship
from sklearn import metrics
import numpy as np
#define true and predicted values
y_true = [100, 50, 70, 60]
y_pred = [98, 50, 68, 60]
# calculate MAE, MSE, RMSE
print(metrics.mean_absolute_error(y_true, y_pred))
print(metrics.mean_squared_error(y_true, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
#Model Evaluation Using Train/Test Split

#replace all categorical variables with numbers
gender = {'male':1,'female':2}
mydata.gender = [gender[item] for item in mydata.gender] 

race={'group A':1,'group B':2,'group C':3,'group D':4,'group E':5} 
mydata.race=[race[item]for item in mydata.race]

lunch={'standard':1,'free/reduced':2}
mydata.lunch=[lunch[item]for item in mydata.lunch]

pre_course={'completed':1,'none':2}
mydata.pre_course=[pre_course[item]for item in mydata.pre_course]

#to replace ' with space
mydata.par_ethnicity=mydata.par_ethnicity.str.replace('[^A-Za-z\s]+', '')
par_ethnicity={'bachelors degree':1,'some college':2,'masters degree':3,'associates degree':4,'high school':5,'some high school':6}
mydata.par_ethnicity=[par_ethnicity[item]for item in mydata.par_ethnicity]


from sklearn.model_selection import train_test_split

X = mydata[['gender','writing_score','reading_score','race','lunch','pre_course','par_ethnicity']]
y = mydata.math_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Instantiate model
lm2 =linear_model.LinearRegression()
# Fit Model
lm2.fit(X_train, y_train)

# Predict
y_pred = lm2.predict(X_test)
#MSE
print((metrics.mean_squared_error(y_test, y_pred)))
# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))