import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()
data=pd.read_csv('../input/101-simple-linear-regressioncsv/1.01. Simple linear regression.csv')

data
data.describe()

#

#SAT=Critical Reading + Mathematics + Writing

#GPA=Grade Point Average

y=data['GPA']

x1=data['SAT']
plt.scatter(x1,y)

plt.xlabel('SAT')

plt.ylabel('GPA')
# y=b0+b1x1



x=sm.add_constant(x1)



results=sm.OLS(y,x).fit()

#Contain Ordinary Least Square Regression 

results.summary()
plt.scatter(x1,y)

yhat=0.2750+0.0017*x1 #yHat=0.275+0.0017x1 Regression Line

fig=plt.plot(x1,yhat, lw=4,c='orange',label='Regression Line')

plt.xlabel('SAT',fontsize='20')

plt.ylabel('GPA',fontsize='20')

plt.show()
raw_data=pd.read_csv('../input/103-dummiescsv/1.03. Dummies.csv')

raw_data
data=raw_data.copy()
#Change Yes=1 , and No = 0

data['Attendance']=data['Attendance'].map({'Yes':1,'No':0})

data
data.describe()
y=data['GPA']

x1=data[['SAT','Attendance']]
x=sm.add_constant(x1)

results=sm.OLS(y,x).fit()

results.summary()
plt.scatter(data['SAT'],y,c=data['Attendance'],cmap='RdYlGn_r')

yHat_no=0.6439+0.0014*data['SAT']

yHat_yes=0.8665+0.0014*data['SAT']

fig=plt.plot(data['SAT'],yHat_no,lw=2,c='#006837')

fig=plt.plot(data['SAT'],yHat_yes,lw=2,c='#a50026')



plt.xlabel('SAT',fontsize=20)

plt.ylabel('GPA',fontsize=20)

plt.show()
x

# Const actually added with the add_constant() method we use prior to fitting the model , it is simulation of x0
new_data=pd.DataFrame({'const':1,'SAT':[1700,1670],'Attendance':[0,1]})

new_data=new_data[['const','SAT','Attendance']]

new_data
#Rename the index 0:Bob and 1:Alice

new_data.rename(index={0:'Bob',1:'Alice'})
#The appropriate method that allow us to predict the values is the fitted regression dot predict

#The fitted regressions for us is variable results , results= sm.OLS(y,x).fit()

predictions = results.predict(new_data)

predictions
#I will transform into a data frame and join it with the first one

predictionsDataFrame=pd.DataFrame({'Predictions':predictions})

joined=new_data.join(predictionsDataFrame)

joined.rename(index={0:'Bob',1:'Alice'})