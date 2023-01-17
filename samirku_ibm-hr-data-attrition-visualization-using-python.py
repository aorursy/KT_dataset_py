#import libraries

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

import warnings    # To suppress warnings
warnings.filterwarnings("ignore")

hrdata = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv", header=0)
hrdata.describe()
#Add columns to store discrete variables for Salary range and alcohol levels

#Adding discrete valriables

hrdata['CmpYrInterval'] = pd.cut(hrdata['YearsAtCompany'], 5, labels=['<9', '<15', '<24', '<32', '<33+'])

hrdata['RateLvl'] = pd.cut(hrdata['DailyRate'], 5, labels=['lvl1', 'lvl2', 'lvl3', 'lvl4', 'lvl5'])

hrdata['PromoYrLvl'] = pd.cut(hrdata['YearsSinceLastPromotion'], 5, labels=['0to3', '3to6', '6to9', '9to12', '12+'])



#Add numneric column for Gender and Attrition columns which are of type objects

hrdata['Gender_num'] = pd.get_dummies(hrdata.Gender, drop_first = True) #drop_first = True -> Whether to get k-1 dummies out

                                                                        #of k categorical levels by removing the first level

hrdata['Attrition_num'] = pd.get_dummies(hrdata.Attrition, drop_first = True)



# Segregating data set based on Attrition value

attr_yes = hrdata[hrdata.Attrition=='Yes'] #subset with Attrition = Yes

attr_no = hrdata[hrdata.Attrition=='No'] ##subset with Attrition = No
#Employee Age distribution

fig, ax1 = plt.subplots(1,2, figsize=(16,4))

attr_yes = hrdata[hrdata.Attrition=='Yes']

sb.distplot(hrdata.Age, ax = ax1[0])

sb.distplot(attr_yes.Age, ax = ax1[1])

plt.ylabel('Attrition = Yes')

plt.show()
# Attrition across Education level and corresponding percentage across the total number of employees

total = hrdata.shape[0] 

hrfig = sb.countplot(x='Education', hue = 'Attrition', data = hrdata)



for p in hrfig.patches:

    height = p.get_height()

    hrfig.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height*100/total),

            ha="center") 

plt.show()
#Attrition across Job Roles

fig, ax6 = plt.subplots(1,2, figsize=(24,6))

sb.countplot(x='Department', hue='Gender', data = attr_yes, ax = ax6[1])

sb.countplot(y='JobRole', hue='Gender', data = attr_yes, ax = ax6[0])

plt.show()
sb.countplot(x='Attrition', hue='MaritalStatus', data=hrdata)

plt.show()
# Attrition count across Daily Rate, Years in company, Years since last promotion and level of stock options

fig, ax2 = plt.subplots(2,2, figsize=(10,10))

sb.countplot(x='RateLvl', data=attr_yes, ax = ax2[0,0])

sb.countplot(x='CmpYrInterval', data=attr_yes, ax = ax2[0,1])

sb.countplot(x='PromoYrLvl', data=attr_yes, ax = ax2[1,0])

sb.countplot(x='StockOptionLevel', data=attr_yes, ax = ax2[1,1])

plt.show()
fig, ax4 = plt.subplots(2,2, figsize=(10,10))

sb.boxplot(x='Attrition', y='DistanceFromHome', data=hrdata, ax = ax4[0,0])

sb.boxplot(x='Attrition', y='PercentSalaryHike', data=hrdata, ax = ax4[0,1])

sb.boxplot(x='Attrition', y='NumCompaniesWorked', data=hrdata, ax = ax4[1,0])

sb.boxplot(x='Attrition', y='MonthlyRate', data=hrdata, ax = ax4[1,1])



plt.show()
#Attrition across Gender and Job Roles

sb.barplot(y='JobRole', x = 'Gender_num', data = attr_yes)

plt.show()
cont_col= ['Attrition', 'Age', 'DistanceFromHome','MonthlyRate','NumCompaniesWorked', 'StockOptionLevel']

sb.pairplot(hrdata[cont_col], kind="reg", diag_kind = "kde", hue = 'Attrition')

plt.show()
sb.jointplot(x='Attrition_num', y='YearsAtCompany', data=hrdata, kind="reg")

sb.jointplot(x='Age', y='MonthlyRate', data=hrdata, kind="kde")

plt.show()
sb.factorplot(x =   'Attrition',     # Categorical

               y =   'Age',          # Continuous

               hue = 'Department',   # Categorical

               col = 'CmpYrInterval',   # Categorical for graph columns

               col_wrap=3,           # Wrap facet after two axes

               kind = 'box',

               data = hrdata)

plt.show()
sb.factorplot(x =   'Attrition',     # Categorical

               y =   'Age',          # Continuous

               hue = 'Department',   # Categorical

               col = 'MaritalStatus',   # Categorical for graph columns

               col_wrap=3,           # Wrap facet after two axes

               kind = 'box',

               data = hrdata)

plt.show()