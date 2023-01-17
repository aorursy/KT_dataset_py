# Call libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt; plt.rcdefaults()

import seaborn as sns

import os

import warnings    # We want to suppress warnings

%matplotlib inline

from matplotlib import colors

warnings.filterwarnings("ignore")    # Ignore warnings

HRData=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

HRData.head()
# Column Datatypes/ null values if any

HRData.info()

HRData.isnull().any()

# There are no missing values. the data is complete
# Find how many fields have unique values to assess their role in data analysis:

hrdunique = HRData.nunique()

hrdunique = hrdunique.sort_values()

hrdunique
# Looking at above unique data list, following columns are irrelevant in analysis so drop the columns where there are single unique values

hrd=HRData.copy()

hrd.drop('Over18', axis=1, inplace=True)

hrd.drop('StandardHours', axis=1, inplace=True)

hrd.drop('EmployeeNumber', axis=1, inplace=True)

hrd.drop('EmployeeCount', axis=1, inplace=True)
# Group the data by Attrition mean to see how average values differ for different parameters

hrd.groupby('Attrition').mean()
# RESULT:

# The average satisfaction level of employees who stayed with the company is higher than that of the employees who left.

# People in same role for longer period stayed with the company longer

# Distance from home also contributed in Attrition as the average here is more for people who left
#Create dataframe for the grouping analysis

df = pd.DataFrame(hrd, columns = ['Gender', 'Department', 'Attrition'])

# group data by department

groupby_DeptnAttrition = df['Gender'].groupby([df['Department'],df['Attrition']])

# view statistics by group

groupby_DeptnAttrition.describe()
num_bins = 30

hrd.hist(bins=num_bins, figsize=(20,15))
pd.crosstab(df.Department,df.Gender).plot(kind='bar')

plt.title('Genderwise employee distribution in all Department')

plt.xlabel('Department')

plt.ylabel('Employee')
# RESULT: More males working in all departments than females
pd.crosstab(hrd.Department,hrd.Attrition).plot(kind='bar')

plt.title('Attrition rate by Department')

plt.xlabel('Department')

plt.ylabel('Frequency of Attrition')
# RESULT: Attrition is highest in Reasearch and Development in comparison to Sales and HR. HR has lowest attrition rate.
sns.distplot(hrd['Age'])
# RESULT: Highest concenteratio of empployees is in the age group of 30 to 40
fig,ax = plt.subplots(1, 2, figsize=(18,4)) 

sns.barplot(x = 'Attrition', y = 'DistanceFromHome', data = hrd, ax = ax[0])

sns.barplot(x = 'Gender',y = 'JobSatisfaction', data = hrd, ax = ax[1])
# RESULT: 

# Attrition increases for the employees having greater distance from home

# JobSatisfaction level in Males are slightly higher in comparison to Females
fig = plt.figure(figsize=(12,12))

sns.distplot(hrd.JobSatisfaction, hist=False, kde=True, label='Job Satisfaction', hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": sns.xkcd_rgb["blue"]})

sns.distplot(hrd.EnvironmentSatisfaction, hist=False, kde=True, label='Environment Satisfaction', hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": sns.xkcd_rgb["red"]})

sns.distplot(hrd.RelationshipSatisfaction, hist=False, kde=True, label='Relationship Satisfaction', hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": sns.xkcd_rgb["green"]})

plt.suptitle('Satisfaction Levels of Employees', size=22, x=0.5, y=0.94)

plt.xlabel('Satisfaction Levels', size=10)

plt.legend()

# RESULT: Frequency of different satisfaction level varies for employees. So no single criteria to impact overall satisfaction among Employees
sns.jointplot(x="Age", y="MonthlyIncome", data=hrd)
# RESULT: Strong co-relation between Age and income. With Age there is substancial increase in income
colors = ('red')

area = np.pi*3

plt.scatter(x="Age", y="YearsAtCompany", data=hrd, c=colors,s=area, alpha=0.5);

plt.title('Scatter plot')

plt.xlabel('Age')

plt.ylabel('YearsAtCompany')

plt.legend(loc=2)
# RESULT: Age group between 30-40 are maximun in no who stick with the company on an average of 5-10 years
fig = plt.figure(figsize=(12,12))

r = sns.heatmap(hrd.corr(), cmap='BuPu',linewidths=.5,annot=True, fmt='.1f')

r.set_title("Heatmap of IBM HR Data")
# RESULT: There are various factors which strongly corelate and impact on overall attriton levels of Employees with various degrees