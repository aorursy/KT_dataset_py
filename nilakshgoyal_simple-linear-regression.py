import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
data = pd.read_csv("../input/random-salary-data-of-employes-age-wise/Salary_Data.csv")
data.head()
data.describe()
##declaring independent and dependent variables
y = data['Salary']
x1 = data['YearsExperience']
##producing a scatter plot to explore the data
plt.scatter(x1,y)
plt.xlabel('YearsExperience',fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.show()
##creating a linear regression
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()
##now plotting the regression line on the scatter plot
plt.scatter(x1,y)
yhat = 9449.96*x1+25790
fig = plt.plot(x1,yhat,lw=2,c='orange')
plt.xlabel('Years of Experience',fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.show()
