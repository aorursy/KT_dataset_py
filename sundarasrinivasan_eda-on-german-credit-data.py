# Loading all the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib notebook

import seaborn as sns

from IPython.display import display, HTML

from IPython.display import display_html

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    for filename in filenames:

        print(os.path.join(dirname, filename))



def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)



print(os.listdir("../input"))

!ls -la
# importing the dataset

df = pd.read_csv('/kaggle/input/germandata.csv')

DF = df.copy()

DF.head(10)

# Head function will give the first 5 rows with all the column values

# Most of the features contains code, we need to work on this
# How many Observations and input feautes

DF.shape
# How many numerical and categorical variables

DF.info()
# Missing value %

# Alternate way to find the missing value % column wise

# Here we dont have missing values

# DF.isna().mean().round(4)*100
DF.describe().T
# Before diving to analysis, We will work on the labels of some variables for easy understanding



# Feature : checking_account_status

# A11: < 0 DM

# A12: 0 <= x < 200 DM

# A13 : >= 200 DM / salary assignments for at least 1 year

# A14 : no checking account

DF['checking_account_status'].replace({'A11':'low','A12':'medium','A13':'high','A14':'none'}, inplace = True)



# Feature : credit_history

# A30 : no credits taken/ all credits paid back duly

# A31 : all credits at this bank paid back duly

# A32 : existing credits paid back duly till now

# A33 : delay in paying off in the past

# A34 : critical account/ other credits existing (not at this bank)

DF['credit_history'].replace({'A30':'No Credits','A31':'Paid Credits','A32':'Existing Credits','A33':'Delay in Past','A34':'Critical'}, inplace = True)



# Feature : purpose

# A40 : car (new)

# A41 : car (used)

# A42 : furniture/equipment

# A43 : radio/television

# A44 : domestic appliances

# A45 : repairs

# A46 : education

# A47 : (vacation - does not exist?)

# A48 : retraining

# A49 : business

# A410 : others

DF['purpose'].replace({'A40':'new car','A41':'used car','A42':'furniture/equipment','A43':'radio/television',\

                        'A44':'domestic appliances','A45':'repairs','A46':'education','A47':'(vacation - does not exist?)',\

                        'A48':'retraining','A49':'business','A410':'others'}, inplace = True)



# Feature : Savings (Balance in savings account)

# A61 : < 100 DM, 

# A62 : 100 <= x < 500 DM

# A63 : 500 <= x < 1000 DM

# A64 : >= 1000 DM

# A65 : unknown/ no savings account

DF['savings'].replace({'A61':'low','A62':'medium','A63':'high','A64':'very high','A65':'no savings'}, inplace = True)



# Feature : Present Employment

# A71 : unemployed

# A72 : < 1 year

# A73 : 1 <= x < 4 years

# A74 : 4 <= x < 7 years

# A75 : .. >= 7 years

DF['present_employment'].replace({'A71':'Unemployed','A72':1,'A73':4,'A74':7,'A75':'7+'}, inplace = True)



# A91 : male : divorced/separated

# A92 : female : divorced/separated/married

# A93 : male : single

# A94 : male : married/widowed

# A95 : female : single

DF['personal'].replace({'A91':'male divorced','A92':'female divorced','A93':'male single','A94':'male married','A95':'female single'}, inplace = True)



# Feature : Other Debtors

# A101 : none

# A102 : co-applicant

# A103 : guarantor

DF['other_debtors'].replace({'A101':'none','A102':'co-applicant','A103':'guarantor'}, inplace = True)



# Feature : Property

# A121 : real estate

# A122 : if not A121 : building society savings agreement/ life insurance

# A123 : if not A121/A122 : car or other, not in attribute 6

# A124 : unknown / no property

DF['property'].replace({'A121':'real estate','A122':'life insurance','A123':'car or other','A124':'no property'}, inplace = True)



# Features : Housing

# A151 : rent

# A152 : own

# A153 : for free

DF['housing'].replace({'A151':'rent','A152':'own','A153':'free'}, inplace = True)



# Feature : Job

# A171 : unemployed/ unskilled - non-resident

# A172 : unskilled - resident

# A173 : skilled employee / official

# A174 : management/ self-employed/highly qualified employee/ officer

DF['job'].replace({'A171':'unskilled - non-resident','A172':'unskilled - resident','A173':'skilled employee','A174':'management/officer'}, inplace = True)

DF['telephone'].replace({'A191':'none','A192':'yes'}, inplace = True)

DF['foreign_worker'].replace({'A201':'yes','A202':'no'}, inplace = True)

DF['other_installment_plans'].replace({'A141':'bank','A142':'stores','A143':'none'}, inplace = True)



# Grouping age

DF['age_group'] = np.nan

DF.loc[(DF['age'] > 18) & (DF['age']<= 29),'age_group'] = 'young'

DF.loc[(DF['age'] > 29) & (DF['age'] <= 45), 'age_group'] = 'middle'

DF.loc[(DF['age'] > 45) & (DF['age'] <= 55), 'age_group'] = 'old'

DF.loc[(DF['age'] > 55),'age_group'] = 'elder'
# Let's start the analysis with some basic questions with Target variable customer type



# What is the data type of target variable ?

DF['customer_type'].dtypes

# int 64



# What are the values in target

DF['customer_type'].unique()

# 1 and 2 are the values in target



# This confirms that we are dealing with Binary classification problem

# 1: Good risk customer

# 2: Bad risk customer
# What is the propotion of genders 



fig,axes = plt.subplots(1,2,figsize = (13,5))

sns.countplot(DF.personal, ax=axes[0])



Gender_wise = DF.groupby(['personal', 'customer_type']).size().reset_index().pivot(columns='customer_type', index='personal', values=0)

Gender_wise.plot(kind = 'bar', stacked = True, ax = axes[1])



display_side_by_side(Gender_wise)
# checking_account_status

fig,axes = plt.subplots(1,2,figsize = (13,5))



sns.countplot(DF.checking_account_status, ax = axes[0])



accountstatus_gender = DF.groupby(['checking_account_status','personal']).size().reset_index().pivot(columns = 'checking_account_status',index = 'personal',values=0)

accountstatus_gender.plot(kind = 'bar', stacked = True, ax = axes[1])

display_side_by_side(accountstatus_gender)
fig,axes = plt.subplots(1,2,figsize = (14,7))



# What is the distribution in duration

sns.boxplot(DF.duration, orient='v', ax = axes[0])



# Whether duration differs gender wise

sns.boxplot(x = DF.personal, y = DF.duration, ax = axes[1])
fig,axes = plt.subplots(1,2,figsize = (14,7))

#  Duration



sns.boxplot(x = DF.personal, y = DF.duration, ax = axes[0])

sns.boxplot(x = DF.personal, y = DF.duration, hue= DF.customer_type, ax = axes[1])
# Credit history



plt.figure(figsize=(7,5))

sns.countplot(DF.credit_history)

print(DF.credit_history.value_counts())



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_CreditHistory = DF.groupby(['personal','credit_history']).size().reset_index().pivot(columns = 'credit_history', index = 'personal', values=0)

Gender_CreditHistory.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.credit_history,hue = DF.customer_type, ax = axes[1])

Gender_CreditHistory
# Purpose

# what are the purposes people tends to apply credits ?

plt.figure(figsize=(15,5))

print(DF['purpose'].value_counts())

sns.countplot(DF.purpose)



# Who purchased what ?

fig,axes = plt.subplots(1,2,figsize=(20,5))

Gender_purpose = DF.groupby(['personal','purpose']).size().reset_index().pivot(columns = 'purpose', index = 'personal', values=0)

Gender_purpose.plot(kind = 'bar', stacked = True, ax = axes[0])

x = sns.countplot(DF.purpose,hue = DF.customer_type, ax = axes[1])

x.set_xticklabels(x.get_xticklabels(), rotation=45)

Gender_purpose
# Credit amount



# what is the distribution of credit amount

sns.distplot(DF.credit_amount)

DF.credit_amount.describe()



# Genderwise credit amount

fig,axes = plt.subplots(1,2,figsize = (14,7))

sns.boxplot(x = DF.personal, y = DF.credit_amount,ax = axes[0])



# Customer type credit amount distribution

sns.boxplot(x = DF.personal, y = DF.credit_amount, hue= DF.customer_type, ax = axes[1])



plt.figure(figsize=(18,5))

x = sns.boxplot(x = DF.purpose, y = DF.credit_amount, hue= DF.customer_type)

x.set_xticklabels(x.get_xticklabels(), rotation=45)



plt.figure(figsize=(18,5))

x = sns.boxplot(x = DF.purpose, y = DF.credit_amount, hue= DF.personal)

x.set_xticklabels(x.get_xticklabels(), rotation=45)



plt.show()
# Savings account



# which savings account is high

sns.countplot(DF.savings)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_savings = DF.groupby(['personal','savings']).size().reset_index().pivot(columns = 'savings', index = 'personal', values=0)

Gender_savings.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.savings,hue = DF.customer_type, ax = axes[1])

Gender_savings
# present_employment

sns.countplot(DF.present_employment)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_experience = DF.groupby(['personal','present_employment']).size().reset_index().pivot(columns = 'present_employment', index = 'personal', values=0)

Gender_experience.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.present_employment,hue = DF.customer_type, ax = axes[1])

Gender_experience
# installment_rate

sns.countplot(DF.installment_rate)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_installment = DF.groupby(['personal','installment_rate']).size().reset_index().pivot(columns = 'installment_rate', index = 'personal', values=0)

Gender_installment.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.installment_rate,hue = DF.customer_type, ax = axes[1])

Gender_installment
# installment_rate

sns.countplot(DF.other_debtors)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_debtors = DF.groupby(['personal','other_debtors']).size().reset_index().pivot(columns = 'other_debtors', index = 'personal', values=0)

Gender_debtors.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.other_debtors,hue = DF.customer_type, ax = axes[1])

Gender_debtors
# present_residence

sns.countplot(DF.present_residence)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_residence = DF.groupby(['personal','present_residence']).size().reset_index().pivot(columns = 'present_residence', index = 'personal', values=0)

Gender_residence.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.present_residence,hue = DF.customer_type, ax = axes[1])

Gender_residence
# property

sns.countplot(DF.property)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_property = DF.groupby(['personal','property']).size().reset_index().pivot(columns = 'property', index = 'personal', values=0)

Gender_property.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.property,hue = DF.customer_type, ax = axes[1])

Gender_property
# other_installment_plans

sns.countplot(DF.other_installment_plans)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_otherinstallment = DF.groupby(['personal','other_installment_plans']).size().reset_index().pivot(columns = 'other_installment_plans', index = 'personal', values=0)

Gender_otherinstallment.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.other_installment_plans,hue = DF.customer_type, ax = axes[1])

Gender_otherinstallment
# housing

sns.countplot(DF.housing)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_housing = DF.groupby(['personal','housing']).size().reset_index().pivot(columns = 'housing', index = 'personal', values=0)

Gender_housing.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.housing,hue = DF.customer_type, ax = axes[1])

Gender_housing
# existing_credits

sns.countplot(DF.existing_credits)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_credits = DF.groupby(['personal','existing_credits']).size().reset_index().pivot(columns = 'existing_credits', index = 'personal', values=0)

Gender_credits.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.existing_credits,hue = DF.customer_type, ax = axes[1])

Gender_credits
# job

plt.figure(figsize=(10,5))

sns.countplot(DF.job)



fig,axes = plt.subplots(1,2,figsize=(17,5))

Gender_job = DF.groupby(['personal','job']).size().reset_index().pivot(columns = 'job', index = 'personal', values=0)

Gender_job.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.job,hue = DF.customer_type, ax = axes[1])

Gender_job
# dependents

plt.figure(figsize=(7,5))

sns.countplot(DF.dependents)



fig,axes = plt.subplots(1,2,figsize=(17,5))

Gender_dependents = DF.groupby(['personal','dependents']).size().reset_index().pivot(columns = 'dependents', index = 'personal', values=0)

Gender_dependents.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.dependents,hue = DF.customer_type, ax = axes[1])

Gender_dependents
# telephone

plt.figure(figsize=(7,5))

sns.countplot(DF.telephone)



fig,axes = plt.subplots(1,2,figsize=(17,5))

Gender_telephone = DF.groupby(['personal','telephone']).size().reset_index().pivot(columns = 'telephone', index = 'personal', values=0)

Gender_telephone.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.telephone,hue = DF.customer_type, ax = axes[1])

Gender_telephone
# foreign_worker

plt.figure(figsize=(7,5))

sns.countplot(DF.foreign_worker)



fig,axes = plt.subplots(1,2,figsize=(17,5))

Gender_foreign = DF.groupby(['personal','foreign_worker']).size().reset_index().pivot(columns = 'foreign_worker', index = 'personal', values=0)

Gender_foreign.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.foreign_worker,hue = DF.customer_type, ax = axes[1])

Gender_foreign
# Age

# property

sns.countplot(DF.age_group)



fig,axes = plt.subplots(1,2,figsize=(15,5))

Gender_age = DF.groupby(['personal','age_group']).size().reset_index().pivot(columns = 'age_group', index = 'personal', values=0)

Gender_age.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.age_group,hue = DF.customer_type, ax = axes[1])

Gender_age

fig,axes = plt.subplots(1,2,figsize=(13,5))



sns.boxplot(x = DF.age_group, y = DF.duration, ax = axes[0])

sns.boxplot(x = DF.age_group, y = DF.duration, hue= DF.customer_type, ax = axes[1])
# Purpose

# what are the purposes people tends to apply credits ?

plt.figure(figsize=(15,5))

print(DF['purpose'].value_counts())

sns.countplot(DF.purpose)



# Who purchased what ?

fig,axes = plt.subplots(1,2,figsize=(20,5))

age_purpose = DF.groupby(['age_group','purpose']).size().reset_index().pivot(columns = 'purpose', index = 'age_group', values=0)

age_purpose.plot(kind = 'bar', stacked = True, ax = axes[0])

x = sns.countplot(DF.purpose,hue = DF.customer_type, ax = axes[1])

x.set_xticklabels(x.get_xticklabels(), rotation=45)

age_purpose
# Genderwise credit amount

fig,axes = plt.subplots(1,2,figsize = (14,7))

sns.boxplot(x = DF.age_group, y = DF.credit_amount,ax = axes[0])



# Customer type credit amount distribution

sns.boxplot(x = DF.age_group, y = DF.credit_amount, hue= DF.customer_type, ax = axes[1])
# present_employment

fig,axes = plt.subplots(1,2,figsize=(15,5))

age_experience = DF.groupby(['age_group','present_employment']).size().reset_index().pivot(columns = 'present_employment', index = 'age_group', values=0)

age_experience.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.present_employment,hue = DF.customer_type, ax = axes[1])

age_experience
# personal

fig,axes = plt.subplots(1,2,figsize=(15,5))

age_gender = DF.groupby(['age_group','personal']).size().reset_index().pivot(columns = 'personal', index = 'age_group', values=0)

age_gender.plot(kind = 'bar', stacked = True, ax = axes[0])

sns.countplot(DF.personal,hue = DF.customer_type, ax = axes[1])

age_gender
# job

fig,axes = plt.subplots(1,2,figsize=(15,5))

age_job = DF.groupby(['age_group','job']).size().reset_index().pivot(columns = 'job', index = 'age_group', values=0)

age_job.plot(kind = 'bar', stacked = True, ax = axes[0])

x = sns.countplot(DF.job,hue = DF.customer_type, ax = axes[1])

x.set_xticklabels(x.get_xticklabels(), rotation=45)

age_job