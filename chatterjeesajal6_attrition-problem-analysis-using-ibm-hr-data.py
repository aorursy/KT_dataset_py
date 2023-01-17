## Call libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import sys



%matplotlib inline

from matplotlib.ticker import NullFormatter  # for plotting muilple distributions with NullFormatter()
## Read and explore data

# Read file and explore dataset

hr_attrition= pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv", header=0)

hr_attrition.columns.values  # Get the column names
#########################################################################################

#  Most of the variables are numerical in the dataset, some columns are actually categorical in nature

hr_attrition.info()
###########################################################################################

# Define a common module for drawing subplots

# This module will draw subplot based on the parameters 

# There will be mutiple subplots within the main plotting window

#  Defination of the parameters are-

#  var_Name - this is the variable name from the data file

#  tittle_Name - this is the Tittle name give for the plot

#  nrow & ncol - this is the number of subplots within the main plotting window

#  idx - position of subplot in the main plotting window

#  fz - the font size of Tittle in the main plotting window

##########################################################################################

def draw_subplots(var_Name,tittle_Name,nrow=1,ncol=1,idx=1,fz=10):

    ax = plt.subplot(nrow,ncol,idx)

    ax.set_title('Distribution of '+var_Name)

    plt.suptitle(tittle_Name, fontsize=fz)



numeric_columns = ['Age', 'MonthlyIncome', 'TotalWorkingYears']



fig,ax = plt.subplots(1,1, figsize=(10,10))

j=0  # reset the counter to plot 

title_Str="Plotting the density distribution of various numeric Features"



for i in numeric_columns:

    j +=1

    draw_subplots(i,title_Str,3,1,j,20) # create a 1x3 subplots for plotting distribution plots

    sns.distplot(hr_attrition[i])

    plt.xlabel('')
numeric_columns = ['Age', 'DistanceFromHome', 'TotalWorkingYears',

                   'YearsAtCompany', 'Education','StockOptionLevel']





fig,ax = plt.subplots(1,1, figsize=(15,15))



j=0 # reset the counter to plot 

title_Str="Plotting the count distributions of various numeric Features"



for i in numeric_columns:

    j +=1

    draw_subplots(i,title_Str,3,2,j,20) # create a 3x2 subplots for plotting distribution plots

    sns.countplot(hr_attrition[i],hue=hr_attrition["Attrition"])

    plt.xlabel('')

numeric_columns = ['Age', 'MonthlyIncome', 'TotalWorkingYears',

                   'YearsAtCompany', 'YearsInCurrentRole','YearsWithCurrManager']



fig,ax = plt.subplots(1,1, figsize=(10,10))

j=0 # reset the counter to plot 

title_Str="Plotting the Boxplot distribution of various numeric Features"



for i in numeric_columns:

    j +=1

    draw_subplots(i,title_Str,3,2,j,20) # create a 3x2 subplots for plotting distribution plots

    sns.boxplot(hr_attrition.Attrition, hr_attrition[i])  # Note the change in bottom level

    plt.xlabel('')

numeric_columns = ['Age','DailyRate','DistanceFromHome','Education','EmployeeCount','EmployeeNumber',

                   'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction',

                   'MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',

                   'RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears',

                   'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',

                   'YearsSinceLastPromotion','YearsWithCurrManager']



# Site :: http://seaborn.pydata.org/examples/many_pairwise_correlations.html

# Compute the correlation matrix

corr=hr_attrition[numeric_columns].corr()



fig,ax = plt.subplots(1,1, figsize=(20,20))

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,yticklabels="auto",xticklabels=False,

            square=True, linewidths=.5,annot=True, fmt= '.1f')

# Look for job satisfaction at various job levels

sns.kdeplot(hr_attrition.JobSatisfaction, hr_attrition.JobInvolvement)

## Job involvement is mostly high with high job satisfaction
# Look for satisfaction with environment with education levels of the employee

sns.kdeplot(hr_attrition.Education, hr_attrition.EnvironmentSatisfaction)

#     Factorplots are plots between one continuous, one categorical

#     conditioned by another one or two categorical variables

sns.factorplot(x =   'Department',

               y =   'Education',

               hue = 'Attrition',

               col=  'BusinessTravel',

               row= 'OverTime',   

               kind = 'box',

               data = hr_attrition)

sns.factorplot(x =   'JobSatisfaction',

               y =   'MonthlyIncome',

               hue = 'Attrition',

               col=  'Education',   

               col_wrap=2,           # Wrap facet after two axes

               kind = 'box',

               data = hr_attrition)

# Distribution of Job roles in pie chart

fig,ax = plt.subplots(1,1, figsize=(10,10))



# The slices will be ordered and plotted counter-clockwise.

labels = hr_attrition['JobRole'].unique()

jr_array = []



for i in range(len(labels)):

    jr_array.append(hr_attrition['JobRole'].value_counts()[i])



plt.pie(jr_array, labels=labels,

                autopct='%1.1f%%', shadow=True, startangle=90)

                # The default startangle is 0, which would start

                # everything is rotated counter-clockwise by 90 degrees,

                # so the plotting starts on the positive y-axis.



plt.title('Job Role Pie Chart', fontsize=20)

plt.show()



sns.factorplot(x =   'WorkLifeBalance',

               y =   'JobRole',

               hue = 'Attrition',

               col=  'PerformanceRating',   

               col_wrap=2,           # Wrap facet after two axes

               kind = 'box',

               data = hr_attrition)
