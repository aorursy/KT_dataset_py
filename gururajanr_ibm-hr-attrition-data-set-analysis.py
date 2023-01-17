#Call libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os
# Read CSV file 

hrdata = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# List the datatype of the columns

hrdata.info()

#look at data

hrdata.head()
# Convert numeric to categoric variables

hrdata['c_Education'] = pd.cut(hrdata['Education'], 5, labels=['Below College','College','Bachelor','Master','Doctor'])

hrdata['c_EnvironmentSatisfaction'] = pd.cut(hrdata['EnvironmentSatisfaction'], 4, labels=['Low','Medium','High','Very High'])

hrdata['c_JobInvolvement'] = pd.cut(hrdata['JobInvolvement'], 4, labels=['Low','Medium','High','Very High'])

hrdata['c_JobSatisfaction'] = pd.cut(hrdata['JobSatisfaction'], 4, labels=['Low','Medium','High','Very High'])

hrdata['c_PerformanceRating'] = pd.cut(hrdata['PerformanceRating'], 4, labels=['Low','Good','Excellent','Outstanding'])

hrdata['c_RelationshipSatisfaction'] = pd.cut(hrdata['RelationshipSatisfaction'], 4, labels=['Low','Medium','High','Very High'])

hrdata['c_WorkLifeBalance'] = pd.cut(hrdata['WorkLifeBalance'], 4, labels=['Bad','Good','Better','Best'])

hrdata['c_Age'] = pd.cut(hrdata['Age'], 4, labels=['Young', 'Middle', 'Senior','Super Senior'])

# Convert Attrition column to numeric - to be used to check correlation

hrdata['n_Attrition']=pd.get_dummies(hrdata.Attrition, drop_first = True)

#List the column names of the dataset

hrdata.columns
sns.distplot(hrdata['Age'])

plt.show()    # Without this command plot will not appear
sns.countplot(hrdata.Attrition)

plt.show()
atr_yes = hrdata[hrdata['Attrition'] == 'Yes']

atr_no = hrdata[hrdata['Attrition'] == 'No']

plt.hist(atr_yes['Age'])

plt.show()
# Scatter Plot 

hrdata.plot(kind='scatter', x='Age', y='DailyRate',alpha = 0.5,color = 'red')

plt.xlabel('Age', fontsize=16)              # label = name of label

plt.ylabel('DailyRate', fontsize=16)

plt.title('Age vs DailyRate Scatter Plot', fontsize=20)            # title = title of plot

#bar plot

sns.barplot(x = hrdata['Gender'], y = atr_yes['JobLevel'])

plt.show()
# Histogram

# bins = number of bar in figure

hrdata.TotalWorkingYears.plot(kind = 'hist',bins = 10,figsize = (15,15))

plt.hist(atr_yes['Department'])

plt.show()
#box plot

sns.boxplot(hrdata['Gender'], hrdata['MonthlyIncome'])

plt.title('MonthlyIncome vs Gender Box Plot', fontsize=20)      

plt.xlabel('MonthlyIncome', fontsize=16)

plt.ylabel('Gender', fontsize=16)

plt.show()

sns.distplot(atr_yes['DistanceFromHome'])

plt.show()

# There is not much realtionship between distance from home and Attrition 
#Count Plot

sns.countplot(hrdata.JobLevel)

plt.title('JobLevel Count Plot', fontsize=20)      

plt.xlabel('JobLevel', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()

# Largest population is between Joblevel 1 and 2
sns.distplot(atr_yes['TotalWorkingYears'])

plt.show()

#  attrition is less as the TotalworkingYears increases and is maximum for 1-10 years 
sns.barplot(x = atr_no['RelationshipSatisfaction'], y = atr_no['EducationField'])

plt.show()
sns.barplot(x = atr_yes['RelationshipSatisfaction'], y = atr_yes['EducationField'])

plt.show()
cont_col= ['DistanceFromHome', 'PerformanceRating','EnvironmentSatisfaction','Attrition','YearsWithCurrManager','NumCompaniesWorked']

sns.pairplot(hrdata[cont_col], kind="reg", diag_kind = "kde" , hue = 'Attrition' )

plt.show()



# Employees falling under the below categories are more likely to quit

# 'PerformanceRating' between 3.0 & 4.0 and 'DistanceFromHome' between 10 & 15

# 'YearsWithCurrManager' less than 5
_ = plt.scatter((hrdata['MonthlyRate'] / hrdata['DailyRate']), hrdata['DailyRate'])

_ = plt.xlabel('Ratio of Monthly to Daily Rate')

_ = plt.ylabel('Daily Rate')

_ = plt.title('Monthly/Daily Rate Ratio vs. Daily Rate')

plt.show()
sns.boxplot(hrdata['Attrition'], hrdata['YearsSinceLastPromotion'])

plt.show()
# pie chart

labels = ['Male', 'Female']

sizes = [hrdata['Gender'].value_counts()[0],

         hrdata['Gender'].value_counts()[1]

        ]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)

ax1.axis('equal')

plt.title('Gender Pie Chart', fontsize=20)

plt.show()

fig = plt.figure(figsize=(12,12))

sns.distplot(hrdata.YearsWithCurrManager, hist=False, kde=True, label='YearsWithCurrManager', hist_kws={"histtype": "step", "linewidth": 3,

                  "alpha": 1, "color": sns.xkcd_rgb["azure"]})

sns.distplot(hrdata.YearsAtCompany, hist=False, kde=True, label='YearsAtCompany', hist_kws={"histtype": "step", "linewidth": 3,

                  "alpha": 1, "color": sns.xkcd_rgb["dark blue green"]})

sns.distplot(hrdata.YearsInCurrentRole, hist=False, kde=True, label='YearsInCurrentRole', hist_kws={"histtype": "step", "linewidth": 3,

                  "alpha": 1, "color": sns.xkcd_rgb["fuchsia"]})

plt.suptitle('Total Years: Current Manager, Role and At Company', size=22, x=0.5, y=0.94)

plt.xlabel('Years', size=16)

plt.ylabel('Count', size=16)

plt.legend(prop={'size':26}, loc=1)

plt.show();
#point plot

sns.pointplot(x="Gender", y="TotalWorkingYears", hue="Attrition", data=hrdata,

              palette={"Yes": "blue", "No": "pink"},

              markers=["*", "o"], linestyles=["-", "--"]);



# Males and Females within the ranges for 'TotalWorkingYears' of 11 to 13 are less likely to quit
#Plot a correlation map for all numeric variables

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(hrdata.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

## Joint scatter plot

sns.jointplot(hrdata.Age,hrdata.DailyRate, kind = "scatter")   

plt.show()
#Joint scatter plot with least square line

sns.jointplot(hrdata.Age,hrdata.DailyRate, kind = "reg")   

plt.show()
## Joint plots with hex bins

sns.jointplot(hrdata.YearsAtCompany,hrdata.YearsInCurrentRole, kind = "hex") 

plt.show()
hrdata['c_YearsAtCompany'] = pd.cut(hrdata['YearsAtCompany'], 5, labels=['<9', '<15', '<24', '<32', '<33+'])

sns.factorplot(x =   'Attrition',     # Categorical

               y =   'Age',          # Continuous

               hue = 'Department',   # Categorical

               col = 'c_YearsAtCompany',   # Categorical for graph columns

               col_wrap=3,           # Wrap facet after two axes

               kind = 'box',

               data = hrdata)

plt.show()

# Attrition is higher in Sales and R&D Departments(Lab Technicican and Research scientist)

# Attrition is higher in lower age group

# Attrition is non existent in HR department in higher age group