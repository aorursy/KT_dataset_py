# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# visulaisation

from matplotlib.pyplot import xticks

%matplotlib inline



# Data display coustomization

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)
data = pd.DataFrame(pd.read_csv('../input/Leads.csv'))

data.head(5) 
#checking duplicates

sum(data.duplicated(subset = 'Prospect ID')) == 0

# No duplicate values
data.shape
data.info()
data.describe()
# As we can observe that there are select values for many column.

#This is because customer did not select any option from the list, hence it shows select.

# Select values are as good as NULL.



# Converting 'Select' values to NaN.

data = data.replace('Select', np.nan)
data.isnull().sum()
round(100*(data.isnull().sum()/len(data.index)), 2)
# # we will drop the columns having more than 70% NA values.

# data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>70)].columns, 1)
# Now we will take care of null values in each column one by one.
# Lead Quality: Indicates the quality of lead based on the data and intuition the the employee who has been assigned to the lead
data['Lead Quality'].describe()
sns.countplot(data['Lead Quality'])
# As Lead quality is based on the intution of employee, so if left blank we can impute 'Not Sure' in NaN safely.

data['Lead Quality'] = data['Lead Quality'].replace(np.nan, 'Not Sure')
sns.countplot(data['Lead Quality'])
# Asymmetrique Activity Index  |

# Asymmetrique Profile Index   \   An index and score assigned to each customer

# Asymmetrique Activity Score  |    based on their activity and their profile

# Asymmetrique Profile Score   \
fig, axs = plt.subplots(2,2, figsize = (10,7.5))

plt1 = sns.countplot(data['Asymmetrique Activity Index'], ax = axs[0,0])

plt2 = sns.boxplot(data['Asymmetrique Activity Score'], ax = axs[0,1])

plt3 = sns.countplot(data['Asymmetrique Profile Index'], ax = axs[1,0])

plt4 = sns.boxplot(data['Asymmetrique Profile Score'], ax = axs[1,1])

plt.tight_layout()
# There is too much variation in thes parameters so its not reliable to impute any value in it. 

# 45% null values means we need to drop these columns.
# data = data.drop(['Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score'],1)
round(100*(data.isnull().sum()/len(data.index)), 2)
# City
data.City.describe()
sns.countplot(data.City)

xticks(rotation = 90)
# Around 60% of the data is Mumbai so we can impute Mumbai in the missing values.
# data['City'] = data['City'].replace(np.nan, 'Mumbai')
# Specailization
data.Specialization.describe()
sns.countplot(data.Specialization)

xticks(rotation = 90)
# It maybe the case that lead has not entered any specialization if his/her option is not availabe on the list,

#  may not have any specialization or is a student.

# Hence we can make a category "Others" for missing values. 
data['Specialization'] = data['Specialization'].replace(np.nan, 'Others')
round(100*(data.isnull().sum()/len(data.index)), 2)
# Tags
data.Tags.describe()
fig, axs = plt.subplots(figsize = (15,7.5))

sns.countplot(data.Tags)

xticks(rotation = 90)
# Blanks in the tag column may be imputed by 'Will revert after reading the email'.
data['Tags'] = data['Tags'].replace(np.nan, 'Will revert after reading the email')
# What matters most to you in choosing a course
data['What matters most to you in choosing a course'].describe()
# Blanks in the this column may be imputed by 'Better Career Prospects'.
# data['What matters most to you in choosing a course'] = data['What matters most to you in choosing a course'].replace(np.nan, 'Better Career Prospects')
# Occupation
data['What is your current occupation'].describe()
# 86% entries are of Unemployed so we can impute "Unemployed" in it.
data['What is your current occupation'] = data['What is your current occupation'].replace(np.nan, 'Unemployed')
# Country
# Country is India for most values so let's impute the same in missing values.

data['Country'] = data['Country'].replace(np.nan, 'India')
round(100*(data.isnull().sum()/len(data.index)), 2)
round(100*(data.isnull().sum()/len(data.index)), 2)
data[["Prospect ID","Lead Number"]].nunique()
print(data.shape)
data["Last Notable Activity"].value_counts()
data.head()
data.shape
data.drop(["Prospect ID"],axis=1).to_csv("Marketing_Leads_India.csv.gz",index=False,compression="gzip")