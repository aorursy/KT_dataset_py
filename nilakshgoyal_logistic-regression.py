##in this we are ploting a curve for estimated salary to purchased and age to purchased

##importing relevent libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()
##loading our data set

raw_data = pd.read_csv('../input/logistic-regression/Social_Network_Ads.csv')

raw_data.head()
##as we are not using the gender and user id we can drop these variables

data = raw_data.drop(['Gender','User ID'], axis=1)

data.head()
##declaring independent and dependent variables

x1 = data['Age']

x2 = data['EstimatedSalary']

y = data['Purchased']
##ploting a scatter plot 

f,(ax1,ax2)=plt.subplots(1,2,figsize=(20,5))

ax1.scatter(data['Age'],data['Purchased'])

ax1.set_title('Age to Purchased')

ax2.scatter(data['EstimatedSalary'],data['Purchased'])

ax2.set_title('Estimated Salary to Purchased')

plt.show()
##plotting a regression line for the dataset

x = sm.add_constant(x1)

reg_line = sm.OLS(y,x)

results_line = reg_line.fit()

plt.scatter(x1,y)

y_hat = x1*results_line.params[1]+results_line.params[0]

plt.plot(x1,y_hat,lw=2.5,c='red')

plt.xlabel('AGE',fontsize=20)

plt.ylabel('Purchased',fontsize=20)

plt.show()

##now creating a logit function curve

reg_log = sm.Logit(y,x)

results_log = reg_log.fit()

def f(x,b0,b1):

    return np.array(np.exp(b0+x*b1) / (1+np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))

x_sorted = np.sort(np.array(x1))



plt.scatter(x1,y)

plt.xlabel('AGE',fontsize=20)

plt.ylabel('Purchased',fontsize=20)

plt.plot(x_sorted,f_sorted,lw=2.5,c='red')

plt.show()
##using the same logic performing on salary to purchase

z = sm.add_constant(x2)

reg_line = sm.OLS(y,z)

results_line = reg_line.fit()

plt.scatter(x2,y)

y_hat = x2*results_line.params[1]+results_line.params[0]

plt.plot(x2,y_hat,lw=2.5,c='red')

plt.xlabel('Salary',fontsize=20)

plt.ylabel('Purchased',fontsize=20)

plt.show()
##now creating a logit function curve

reg_log = sm.Logit(y,z)

results_log = reg_log.fit()

def f(z,c0,c1):

    return np.array(np.exp(c0+z*c1) / (1+np.exp(c0+z*c1)))

f_sorted1 = np.sort(f(x2,results_log.params[0],results_log.params[1]))

x_sorted1 = np.sort(np.array(x2))



plt.scatter(x2,y)

plt.xlabel('Salary',fontsize=20)

plt.ylabel('Purchased',fontsize=20)

plt.plot(x_sorted1,f_sorted1,lw=2.5,c='red')

plt.show()