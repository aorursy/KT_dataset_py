

#Call the required libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings    # We want to suppress warnings

import os
#Read Data







IBM_hrdata = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

IBM_hrdata.info()

IBM_hrdata.head()

#Check the unique columns in the data to find which are categorical





Nunique = IBM_hrdata.nunique()

Nunique = Nunique.sort_values()

Nunique


#Distribution Plots for Age, Total Working Years, Years at Company



sns.distplot(IBM_hrdata['Age'])

plt.show() 



sns.distplot(IBM_hrdata['TotalWorkingYears'])

plt.show() 



sns.distplot(IBM_hrdata['YearsAtCompany'])

plt.show() 

#Multiple Distribution Plots 



#Age, Job role, Monthly income, Daily Rate, Hourly Rate, Percentage Saalry Hike, Total Working Years, Years at company, Years since Last promotion, Years with current manager, Number of companies worked, Years in Current Role 







fig, ax = plt.subplots(figsize=(10,10), ncols=4, nrows=3) 

sns.distplot(IBM_hrdata['Age'], ax = ax[0,0]) 

sns.distplot(IBM_hrdata['JobLevel'], ax = ax[0,1]) 

sns.distplot(IBM_hrdata['MonthlyIncome'], ax = ax[0,2]) 

sns.distplot(IBM_hrdata['DailyRate'], ax = ax[0,3]) 

sns.distplot(IBM_hrdata['HourlyRate'], ax = ax[1,0]) 

sns.distplot(IBM_hrdata['PercentSalaryHike'], ax = ax[1,1])

sns.distplot(IBM_hrdata['TotalWorkingYears'], ax = ax[1,2]) 

sns.distplot(IBM_hrdata['YearsAtCompany'], ax = ax[1,3])

sns.distplot(IBM_hrdata['YearsSinceLastPromotion'], ax = ax[2,0]) 

sns.distplot(IBM_hrdata['NumCompaniesWorked'], ax = ax[2,1]) 

sns.distplot(IBM_hrdata['YearsInCurrentRole'], ax = ax[2,2]) 

sns.distplot(IBM_hrdata['YearsWithCurrManager'], ax = ax[2,3]) 

plt.show()
#Count Plot



#Age vs attrition





sns.countplot(x=IBM_hrdata['Age'], hue="Attrition", data=IBM_hrdata) 

plt.xticks( rotation=90)

plt.show()



#Multiple Count plots

#Marital status,Performace rating, Job Level,   Relationship Satisfaction, Over Time, Department, Environmental Satisfaction , job Satisfaction, Work life Balace









# create a data frame to store all required columns



data_df1=pd.DataFrame(IBM_hrdata[['MaritalStatus','PerformanceRating','JobLevel','RelationshipSatisfaction','OverTime','Department','EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance']])



for i, col in enumerate(data_df1.columns):

	sns.countplot(x=col, hue='Attrition',data=IBM_hrdata)	

	plt.figure(i)



#Multiple bar plots



# create a data frame to store all required columns



data_df2=pd.DataFrame(IBM_hrdata[['PerformanceRating','JobSatisfaction','WorkLifeBalance']])



for i, col in enumerate(data_df2.columns):

    sns.barplot(x=col,y='JobLevel', hue='Attrition',data=IBM_hrdata)

    plt.figure(i)

   





#Box Plot



sns.boxplot(IBM_hrdata['JobRole'], IBM_hrdata['JobInvolvement'])

plt.xticks( rotation=90)

plt.show()

sns.boxplot(IBM_hrdata['JobRole'], IBM_hrdata['JobSatisfaction'])

plt.xticks( rotation=90)

plt.show()

sns.boxplot(IBM_hrdata['Gender'], IBM_hrdata['JobSatisfaction'])

plt.xticks( rotation=90)

plt.show()

#sns.boxplot(IBM_hrdata['Gender'], IBM_hrdata['BusinessTravel'])

#plt.show()

#sns.boxplot(IBM_hrdata['Joblevel'], IBM_hrdata['BusinessTravel'])

#plt.show()



#Swarm Plot



sns.swarmplot(IBM_hrdata['JobRole'],IBM_hrdata['MonthlyIncome'])

plt.xticks( rotation=90)

plt.show()

sns.swarmplot(IBM_hrdata['Department'],IBM_hrdata['MonthlyIncome'])

plt.xticks( rotation=90)

plt.show()

sns.swarmplot(IBM_hrdata['EducationField'],IBM_hrdata['MonthlyIncome'])

plt.xticks( rotation=90)

plt.show()
# joint Plots

sns.jointplot(IBM_hrdata.Age,IBM_hrdata.MonthlyIncome, kind = "hex")   

plt.show()
#Pair Plots

cont_col= ['Attrition','JobLevel','Age','EducationField','MonthlyIncome']

sns.pairplot(IBM_hrdata[cont_col],  kind="reg", diag_kind = "kde"  , hue = 'Attrition' )

plt.show()
#Factor plots

sns.factorplot(x =   'Attrition',

               y =   'MonthlyIncome',

               hue = 'MaritalStatus',

               col=  'JobLevel',   

               col_wrap=2,           # Wrap facet after two axes

               kind = 'box',

               data = IBM_hrdata)

plt.show()