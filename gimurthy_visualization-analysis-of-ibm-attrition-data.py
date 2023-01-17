# Load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings    

import os



# Ignore warnings

warnings.filterwarnings("ignore")   
# Load the dataset

ibm_data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv', header=0)

ibm_data.head(5)
# Check if there are any NaN values in the dataset

ibm_data.isnull().any().any()
# Analyze the unique values in the Dataset



# Get the column names of the dataset

cols = ibm_data.columns

# Print the uniques values of each of the columns in the dataset

for col in cols:

    print(col, "   ::  [", np.unique(ibm_data[col]).size,"]")
# Remove the irrelevant columns from the dataset

ibm_data = ibm_data.drop(['EmployeeCount','Over18','StandardHours'], axis=1)
# Convert Factors into Integer values to draw correlation and some charts...

# Not sure if there is a library or alternate available like Caret in R for achieving this.

ibm_data_num = ibm_data

ibm_data_num=ibm_data_num.replace({'Attrition':{'No':0,'Yes':1}})

ibm_data_num=ibm_data_num.replace({'Gender':{'Male':0,'Female':1}})

ibm_data_num=ibm_data_num.replace({'BusinessTravel':{'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2}})

ibm_data_num=ibm_data_num.replace({'Department':{'Human Resources':0,'Research & Development':1,'Sales':2}})

ibm_data_num=ibm_data_num.replace({'EducationField':{'Human Resources':0,'Life Sciences':1,'Marketing':2,'Medical':3,'Technical Degree':4,'Other':5}})

ibm_data_num=ibm_data_num.replace({'JobRole':{'Human Resources':0,'Healthcare Representative':1,'Laboratory Technician':2,'Manager':3,'Manufacturing Director':4,'Research Director':5,'Research Scientist':6,'Sales Executive':7,'Sales Representative':8}})

ibm_data_num=ibm_data_num.replace({'MaritalStatus':{'Single':0,'Married':1,'Divorced':2}}) 

ibm_data_num=ibm_data_num.replace({'OverTime':{'No':0,'Yes':1}})
# Plot the HeatMap for the correlation of the columns



sns.set(style="white")



# Compute the correlation data based on the numerized dataset

ibm_data_corr = ibm_data_num.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(ibm_data_corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(ibm_data_corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Building a dataset for further charting considering only the attributes that have some kind of correlation with Attrition data

ibm_data_temp = ibm_data_num[['Attrition', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'MaritalStatus','MonthlyIncome', 'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']]

ibm_data_temp.head(3)
row = 5

column = 3

x = 0

y = 0



fig,ax = plt.subplots(row, column, figsize=(15,20)) 



for i in range(ibm_data_temp.shape[1]):

   

    if ibm_data_temp.columns.values[i] != 'Attrition':

        sns.boxplot(x='Attrition', y=ibm_data_temp.columns.values[i], data=ibm_data_temp, hue='Attrition', ax = ax[x,y])

        y = y + 1

    

    if y == column:

        x = x + 1

        y = 0

        

    if x*y == ibm_data_temp.shape[1] - 1:

        break

    

plt.legend

plt.show()
# Get the subset of data of only Attrition cases

attr_data = ibm_data.query("Attrition == 'Yes'")

attr_data.head(3)
fig,ax = plt.subplots(1, 3, figsize=(18,4)) 



sns.countplot(x='BusinessTravel',data=attr_data, ax = ax[0])

sns.countplot(x='Department',data=attr_data, ax = ax[1])

sns.countplot(x='EducationField',data=attr_data, ax = ax[2])
plt.figure(figsize=(20,10))

sns.countplot(x='JobRole',data=ibm_data, hue='Attrition')
plt.figure(figsize=(10,5))

sns.countplot(x='Gender',data=ibm_data, hue='Attrition')
plt.figure(figsize=(10,5))

sns.countplot(x='MaritalStatus',data=ibm_data, hue='Attrition')
sns.distplot(ibm_data.query("Attrition == 'Yes'").Age, kde=True, label='Yes Attrition')

sns.distplot(ibm_data.query("Attrition == 'No'").Age, kde=True, label='No Attrition')

plt.title('Age Distribution plot', fontsize=20)  

plt.legend(prop={'size':15}, loc=1)

plt.show()
fig,ax = plt.subplots(1, 2, figsize=(15,4)) 

sns.distplot(attr_data.MonthlyIncome, kde=True, hist=False, label='Montly Income', ax=ax[0])

sns.distplot(attr_data.TotalWorkingYears, kde=True, hist=False, label='Total Expereince', ax=ax[1])

plt.legend(loc=1)

plt.show()