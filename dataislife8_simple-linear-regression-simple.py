import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set() #SETTING MATPLOTLIB WITH SEABORN 
#FROM DEVICE -> data = pd.read_csv(r'S:\U1.01. Simple linear regression.csv')

#From KAGGLE :-

data = pd.read_csv("../input/1.01. Simple linear regression.csv")
data.head() #to take glimpse of the data
y = data['GPA']     # -> Dependent variable (TO BE PREDICTED)

x1 = data['SAT']    # -> Independent variable
plt.plot(x1,y)

plt.xlabel('SAT')

plt.ylabel('GPA')

plt.show()

#This graph provides us about the idea that whether a line can fit or not.
x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
#In the above summary we can see that we get the COEFFICIENT of CONSTANT VARIABLE and our INDEPENDENT VARIABLE x that is SAT
plt.scatter(x1,y)

yhat = 0.0017*x1 + 0.2750   # With the help of coeff we get the equation

# It can be seen as -> GPA = 0.2750 + 0.0017 * SAT or y = 0.2750 + 0.0017 * x



plt.plot(x1,yhat, c='red', lw=4)

plt.xlabel('SAT')

plt.ylabel('GPA')

plt.show()



# Now you can take any value of x to predict y and test the prediction manually.