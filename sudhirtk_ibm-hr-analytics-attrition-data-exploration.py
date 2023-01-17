import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os
empatt = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

empatt.head()
# lets plot the data and look at it, ex: find the Age distribution of ppl in the dataset

sns.distplot(empatt['Age'], color = 'red')

plt.show()
# Display multiple distribution plots.

fig,ax = plt.subplots(3,3, figsize=(10,10))

sns.distplot(empatt['YearsAtCompany'], ax = ax[0,0], color = 'cyan')   

sns.distplot(empatt['YearsInCurrentRole'], ax = ax[0,1],color = 'cyan')

sns.distplot(empatt['YearsSinceLastPromotion'], ax = ax[0,2],color = 'cyan')   

sns.distplot(empatt['YearsWithCurrManager'], ax = ax[1,0], color = 'cyan')

sns.distplot(empatt['TotalWorkingYears'], ax = ax[1,1], color = 'cyan')  

sns.distplot(empatt['TrainingTimesLastYear'], ax = ax[1,2], color = 'cyan')  

sns.distplot(empatt['WorkLifeBalance'], ax = ax[2,0], color = 'cyan')  

sns.distplot(empatt['DistanceFromHome'], ax = ax[2,1], color = 'cyan')  

sns.distplot(empatt['RelationshipSatisfaction'], ax = ax[2,2], color = 'cyan')  

plt.show()
sns.countplot(empatt['Attrition'])

plt.show()
#Bar plot of Job Satisfaction with respect to distance from home according to gender

#Estimator used is median

from numpy import median

sns.barplot(x='Attrition', y='MonthlyIncome', hue= 'Gender',data=empatt, estimator=median)

plt.show()
# Box Plot

f,ax = plt.subplots(figsize=(15,10))

sns.boxplot(x='JobSatisfaction', y='MonthlyIncome', data=empatt, hue='JobLevel',palette='Set3')

plt.legend(loc='best')

plt.show()
## Joint scatter plot

sns.jointplot(empatt.Age,empatt.MonthlyIncome, kind = "scatter")   

plt.show()
g = sns.FacetGrid(empatt, col="JobSatisfaction", row="Gender")

g.map(sns.kdeplot, "MonthlyIncome", "YearsInCurrentRole")

plt.show()
# Pair Plots

data1 = ['Attrition','Age','MonthlyIncome','DistanceFromHome']

sns.pairplot(empatt[data1], kind="reg", diag_kind = "kde" , hue = 'Attrition' )

plt.show()
data2 = ['Gender', 'HourlyRate','DailyRate','MonthlyRate','PercentSalaryHike']

sns.pairplot(empatt[data2], kind="reg", diag_kind = "kde" , hue = 'Gender' )

plt.show()
#Plot a correlation map for all numeric variables

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(empatt.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
total_records= len(empatt)

columns = ["Gender","MaritalStatus","WorkLifeBalance","EnvironmentSatisfaction","JobSatisfaction",

           "JobLevel","BusinessTravel","Department"]



j=0

for i in columns:

    j +=1

    plt.subplot(4,2,j)

    ax1 = sns.countplot(data=empatt,x= i,hue="Attrition")

    if(j==8 or j== 7):

        plt.xticks( rotation=90)

    for p in ax1.patches:

        height = p.get_height()

        ax1.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}'.format(height/total_records,0),

                ha="center",rotation=0) 



# Custom the subplot layout

plt.subplots_adjust(bottom=-0.9, top=2)

plt.show()