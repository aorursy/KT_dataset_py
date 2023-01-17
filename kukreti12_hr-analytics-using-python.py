import numpy as np # Numerical Python
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), easy to use for data structures
import seaborn as sns # Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import warnings # Warning messages are typically issued in situations where it is useful to alert the user of some condition in a program
import os# This module provides a portable way of using operating system dependent functionality
import matplotlib.pyplot as plt# Visualization 
warnings.filterwarnings("ignore") #Never print the matching warning
hr = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv", header=0) #Read the dataset
hr.head() #Display first 5 rows of the data
# Reference for adding title for each sub plots: https://gist.github.com/dyerrington/dac39db54161dafc9359995924413a12
fig,ax = plt.subplots(2,2, figsize=(10,10))               # 'ax' has references to all the four axes
plt.suptitle("Understanding the distribution of various factors", fontsize=20)
sns.distplot(hr['Age'], ax = ax[0,0])  # Plot on 1st axes
ax[0][0].set_title('Distribution of Age')
sns.distplot(hr['TotalWorkingYears'], ax = ax[0,1])  # Plot on IInd axes
ax[0][1].set_title('Distribution of Total Working Years')
sns.distplot(hr['YearsAtCompany'], ax = ax[1,0])  # Plot on IIIrd axes
ax[1][0].set_title('Distribution of Years at company')
sns.distplot(hr['YearsInCurrentRole'], ax = ax[1,1])  # Plot on IV the axes
ax[1][1].set_title('Distribution of Years in Current Role')
plt.show()                                                # Show all of them now
sns.countplot(hr['Attrition'])
plt.show()
#Bar plot of MOnthly income from the rate of attrition according to gender
#Estimator used is median
from numpy import median
sns.barplot(x='Attrition', y='MonthlyIncome', hue= 'Gender',data=hr, estimator=median)# hue= Categorical variable , estimator- Statistical function to estimate within each categorical bin.
plt.show()
#Bar plot of Job Satisfaction with respect to Monthly income from home according to gender
#Estimator used is median
from numpy import median
sns.barplot(x='Attrition', y='MonthlyIncome', hue= 'JobSatisfaction',data=hr, estimator=median)# hue= Categorical variable , estimator- Statistical function to estimate within each categorical bin.
plt.show()
#Bar plot of Job Satisfaction with respect to Monthly income from home according to gender
#Estimator used is median
from numpy import median
sns.barplot(x='Attrition', y='MonthlyIncome', hue= 'MaritalStatus',data=hr, estimator=median)# hue= Categorical variable , estimator- Statistical function to estimate within each categorical bin.
plt.show()
f,ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='JobSatisfaction', y='MonthlyIncome', data=hr, hue='JobLevel',palette='Set3')
plt.legend(loc='best')
plt.show()
sns.violinplot(x="Attrition", y="YearsAtCompany", hue="Gender", data=hr, palette="muted", split=True,
               inner="stick")# When using hue nesting with a variable that takes two levels,
                                 # setting split to True will draw half of a violin for each level. 
                                #This can make it easier to directly compare the distributions.
plt.show()
## Joint scatter plot
sns.jointplot(hr.Age,hr.MonthlyIncome, kind = "scatter")   
plt.show()
hr['age_group'] = pd.cut(hr['Age'], 3, labels=['Young', 'Middle', 'Senior']) #Slicing the continuous data into various groups 
                                                                             #Age Group is the name of new column
sns.factorplot(x =   'Attrition',     # Categorical
               y =   'MonthlyIncome',      # Continuous
               hue = 'JobLevel',    # Categorical
               col = 'age_group',
               col_wrap=2,           # Wrap facet after two axes
               kind = 'box',
               data = hr)
plt.show()
g = sns.FacetGrid(hr, col="JobSatisfaction", row="Gender")
g.map(sns.kdeplot, "MonthlyIncome", "YearsInCurrentRole")
plt.show()
data1 = ['Attrition','Age','MonthlyIncome','DistanceFromHome']
sns.pairplot(hr[data1], kind="reg", diag_kind = "kde" , hue = 'Attrition' )
plt.show()
data2 = ['Gender', 'HourlyRate','DailyRate','MonthlyRate','PercentSalaryHike']
sns.pairplot(hr[data2], kind="reg", diag_kind = "kde" , hue = 'Gender' )
plt.show()
#Plot a correlation map for all numeric variables
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(hr.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)
plt.show()