#Call libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from statsmodels.graphics.mosaicplot import mosaic

import os
#read data

ibm = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv", header = 0)
#top rows

ibm.head()
atr_cnt = ibm["Attrition"].value_counts()



sns.countplot(atr_cnt)

plt.show()
#mosaic plot

ibm.Age.max() #Max age

ibm.Age.min() #Min age



ibm['cat_age'] = pd.cut(ibm['Age'], 3, labels=['young', 'middle', 'old']) #Create categorical column for age grouped by low/middle/high

ibm.head()



from statsmodels.graphics.mosaicplot import mosaic

plt.rcParams['font.size'] = 12.0

mosaic(ibm, ['cat_age', 'Attrition'])

plt.show()

#barplot

fig,ax = plt.subplots(2,2, figsize=(10,10))                       # 'ax' has references to all the four axes

sns.boxplot(ibm['Attrition'], ibm['DailyRate'], ax = ax[0,0])  # Plot on 1st axes 

sns.boxplot(ibm['Attrition'], ibm['HourlyRate'], ax = ax[0,1])  # Plot on IInd axes

sns.boxplot(ibm['Attrition'], ibm['MonthlyRate'], ax = ax[1,0])       # Plot on IIIrd axes

sns.boxplot(ibm['Attrition'], ibm['MonthlyIncome'], ax = ax[1,1])       # Plot on IVth axes

plt.show()
#crosstable

ibm_cc = pd.crosstab(index=ibm["Attrition"], 

                          columns=ibm["Department"])





ibm_cc.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True)

plt.show()
#two-way table

grouped = ibm.groupby(['Attrition','Gender'])

gr = grouped.size()



gr.plot( kind = "line",

figsize=(8,8))

plt.show()
#facet grid

g=sns.FacetGrid(ibm,row='JobRole',col='JobLevel',size=2.2,aspect=1.6)

g.map(plt.hist,'Attrition')

g.add_legend()

plt.show()
# transform overtime from categorical to numeric



number = LabelEncoder()

ibm['OverTime_num'] = number.fit_transform(ibm['OverTime'].astype('str'))



no = ibm[ibm['Attrition'] == 'No']

yes = ibm[ibm['Attrition'] == 'Yes']



fig,ax = plt.subplots(2,2, figsize=(10,10))  # 'ax' has references to all the four axes

sns.barplot(no['OverTime_num'], ax = ax[0,0])  # Plot on 1st axes 

sns.barplot(no['StandardHours'], ax = ax[0,1])  # Plot on IInd axes

sns.barplot(yes['OverTime_num'], ax = ax[1,0])       # Plot on IIIrd axes

sns.barplot(yes['StandardHours'], ax = ax[1,1])       # Plot on IVth axes

plt.show()
#dist plot



no = ibm[ibm['Attrition'] == 'No']

yes = ibm[ibm['Attrition'] == 'Yes']





fig,ax = plt.subplots(2,2, figsize=(10,10))  # 'ax' has references to all the four axes

sns.distplot(no['PercentSalaryHike'], ax = ax[0,0])  # Plot on 1st axes 

sns.distplot(no['PerformanceRating'], ax = ax[0,1])  # Plot on IInd axes

sns.distplot(yes['PercentSalaryHike'], ax = ax[1,0])       # Plot on IIIrd axes

sns.distplot(yes['PerformanceRating'], ax = ax[1,1])       # Plot on IVth axes

plt.show()
#count plot



fig,ax = plt.subplots(2,2, figsize=(10,10))  # 'ax' has references to all the four axes

sns.countplot(no['TotalWorkingYears'], ax = ax[0,0])  # Plot on 1st axes 

sns.countplot(no['YearsInCurrentRole'], ax = ax[0,1])  # Plot on IInd axes

sns.countplot(yes['TotalWorkingYears'], ax = ax[1,0])       # Plot on IIIrd axes

sns.countplot(yes['YearsInCurrentRole'], ax = ax[1,1])       # Plot on IVth axes

plt.show()