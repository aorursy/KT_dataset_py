#Call libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings    # We want to suppress warnings

import os
hrdata = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

hrdata.info()

hrdata.head(10)

Nunique = hrdata.nunique()

Nunique = Nunique.sort_values()

Nunique
sns.distplot(hrdata['Age'])

plt.show() 
#  Plot areas are called axes



fig,ax = plt.subplots(3,3, figsize=(10,10))               # 'ax' has references to all the four axes

sns.distplot(hrdata['TotalWorkingYears'], ax = ax[0,0]) 

sns.distplot(hrdata['YearsAtCompany'], ax = ax[0,1]) 

sns.distplot(hrdata['DistanceFromHome'], ax = ax[0,2]) 

sns.distplot(hrdata['YearsInCurrentRole'], ax = ax[1,0]) 

sns.distplot(hrdata['YearsWithCurrManager'], ax = ax[1,1]) 

sns.distplot(hrdata['YearsSinceLastPromotion'], ax = ax[1,2]) 

sns.distplot(hrdata['PercentSalaryHike'], ax = ax[2,0]) 

sns.distplot(hrdata['YearsSinceLastPromotion'], ax = ax[2,1]) 

sns.distplot(hrdata['TrainingTimesLastYear'], ax = ax[2,2]) 

plt.show()
total_records= len(hrdata)

columns = ["Gender","MaritalStatus","WorkLifeBalance","EnvironmentSatisfaction","JobSatisfaction",

           "JobLevel","BusinessTravel","Department"]



j=0

for i in columns:

    j +=1

    plt.subplot(4,2,j)

    #sns.countplot(hrdata[i])

    ax1 = sns.countplot(data=hrdata,x= i,hue="Attrition")

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

# MaritalStatus wise

columns = ["DistanceFromHome",

"WorkLifeBalance"]



j=0

for i in columns:

    j +=1

    plt.subplot(1,2,j)

    sns.barplot(x = 'Attrition', y = hrdata[i], hue="MaritalStatus", data =hrdata)



#plt.subplots_adjust(bottom=-0.9, top=2)



plt.show()



#JobLevel wise

columns = ["DistanceFromHome",

"WorkLifeBalance",

"PercentSalaryHike"]



j=0

for i in columns:

    j +=1

    plt.subplot(3,1,j)

    sns.barplot(x = 'Attrition', y = hrdata[i], hue="JobLevel", data =hrdata)



plt.subplots_adjust(bottom=-0.9, top=2)



plt.show()



# Display multiple box plots.

#  Plot areas are called axes



fig,ax = plt.subplots(2,2, figsize=(10,10))                       # 'ax' has references to all the four axes

sns.boxplot(hrdata['Attrition'], hrdata['MonthlyIncome'], ax = ax[0,0])  # Plot on 1st axes 

sns.boxplot(hrdata['Gender'], hrdata['MonthlyIncome'], ax = ax[0,1])  # Plot on IInd axes

plt.xticks( rotation=90)

sns.boxplot(hrdata['Department'], hrdata['MonthlyIncome'], ax = ax[1,0])       # Plot on IIIrd axes

plt.xticks( rotation=90)



sns.boxplot(hrdata['JobRole'], hrdata['MonthlyIncome'], ax = ax[1,1])     # Plot on IV the axes

plt.show() 



sns.swarmplot(x="Department", y="MonthlyIncome", hue="Attrition", data=hrdata);

plt.show()



sns.swarmplot(x="JobRole", y="MonthlyIncome", hue="Attrition", data=hrdata);

plt.xticks( rotation=90 )

plt.show()





sns.swarmplot(x="JobLevel", y="MonthlyIncome", hue="Attrition", data=hrdata);

plt.show()
sns.factorplot(x =   'Department',     # Categorical

               y =   'MonthlyIncome',      # Continuous

               hue = 'Attrition',    # Categorical

               col = 'JobLevel',

               col_wrap=2,           # Wrap facet after two axes

               kind = 'swarm',

               data = hrdata)

plt.xticks( rotation=90 )

plt.show()



#Plot a correlation map for all numeric variables

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(hrdata.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

fig,ax = plt.subplots(figsize=(10,10))

sns.violinplot(x = 'Gender',y = 'MonthlyIncome',data=hrdata, hue='Attrition',split=True,palette='Set3')

plt.legend(loc='best')

plt.show()
## Joint scatter plot

sns.jointplot(hrdata.Age,hrdata.MonthlyIncome, kind = "scatter")   

plt.show()



#Joint scatter plot with least square line

sns.jointplot(hrdata.TotalWorkingYears,hrdata.MonthlyIncome, kind = "reg")   

plt.show()



cont_col= ['Attrition','Age','MonthlyIncome', 'JobLevel','DistanceFromHome']

sns.pairplot(hrdata[cont_col],  kind="reg", diag_kind = "kde"  , hue = 'Attrition' )

plt.show()
cont_col= ['Attrition','JobLevel','TotalWorkingYears', 'PercentSalaryHike','PerformanceRating']

sns.pairplot(hrdata[cont_col], kind="reg", diag_kind = "kde" , hue = 'Attrition' )

plt.show()