import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



loan = pd.read_csv("../input/loan-analysis-data/loan.csv", sep=",")

loan.info()
# let's look at the first few rows of the df

loan.head()
# Looking at all the column names

loan.columns
# summarising number of missing values in each column

loan.isnull().sum()
# percentage of missing values in each column

round(loan.isnull().sum()/len(loan.index), 2)*100
# removing the columns having more than 90% missing values

missing_columns = loan.columns[100*(loan.isnull().sum()/len(loan.index)) > 90]

print(missing_columns)
loan = loan.drop(missing_columns, axis=1)

print(loan.shape)



# summarise number of missing values again

100*(loan.isnull().sum()/len(loan.index))
# There are now 2 columns having approx 32 and 64% missing values - 

# description and months since last delinquent



# let's have a look at a few entries in the columns

loan.loc[:, ['desc', 'mths_since_last_delinq']].head()
# dropping the two columns

loan = loan.drop(['desc', 'mths_since_last_delinq'], axis=1)
# summarise number of missing values again

100*(loan.isnull().sum()/len(loan.index))
# missing values in rows

loan.isnull().sum(axis=1)
# checking whether some rows have more than 5 missing values

len(loan[loan.isnull().sum(axis=1) > 5].index)
loan.info()
# The column int_rate is character type, let's convert it to float

loan['int_rate'] = loan['int_rate'].apply(lambda x: pd.to_numeric(x.split("%")[0]))
# checking the data types

loan.info()
# also, lets extract the numeric part from the variable employment length



# first, let's drop the missing values from the column (otherwise the regex code below throws error)

loan = loan[~loan['emp_length'].isnull()]



# using regular expression to extract numeric values from the string

import re

loan['emp_length'] = loan['emp_length'].apply(lambda x: re.findall('\d+', str(x))[0])



# convert to numeric

loan['emp_length'] = loan['emp_length'].apply(lambda x: pd.to_numeric(x))
# looking at type of the columns again

loan.info()
behaviour_var =  [

  "delinq_2yrs",

  "earliest_cr_line",

  "inq_last_6mths",

  "open_acc",

  "pub_rec",

  "revol_bal",

  "revol_util",

  "total_acc",

  "out_prncp",

  "out_prncp_inv",

  "total_pymnt",

  "total_pymnt_inv",

  "total_rec_prncp",

  "total_rec_int",

  "total_rec_late_fee",

  "recoveries",

  "collection_recovery_fee",

  "last_pymnt_d",

  "last_pymnt_amnt",

  "last_credit_pull_d",

  "application_type"]

behaviour_var
# let's now remove the behaviour variables from analysis

df = loan.drop(behaviour_var, axis=1)

df.info()
# also, we will not be able to use the variables zip code, address, state etc.

# the variable 'title' is derived from the variable 'purpose'

# thus let get rid of all these variables as well



df = df.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)
df['loan_status'] = df['loan_status'].astype('category')

df['loan_status'].value_counts()
# filtering only fully paid or charged-off

df = df[df['loan_status'] != 'Current']

df['loan_status'] = df['loan_status'].apply(lambda x: 0 if x=='Fully Paid' else 1)



# converting loan_status to integer type

df['loan_status'] = df['loan_status'].apply(lambda x: pd.to_numeric(x))



# summarising the values

df['loan_status'].value_counts()
# default rate

round(np.mean(df['loan_status']), 2)
# plotting default rates across grade of the loan

sns.barplot(x='grade', y='loan_status', data=df)

plt.show()
# lets define a function to plot loan_status across categorical variables

def plot_cat(cat_var):

    sns.barplot(x=cat_var, y='loan_status', data=df)

    plt.show()

    
# compare default rates across grade of loan

plot_cat('grade')
# term: 60 months loans default more than 36 months loans

plot_cat('term')
# sub-grade: as expected - A1 is better than A2 better than A3 and so on 

plt.figure(figsize=(16, 6))

plot_cat('sub_grade')
# home ownership: not a great discriminator

plot_cat('home_ownership')
# verification_status: surprisingly, verified loans default more than not verifiedb

plot_cat('verification_status')
# purpose: small business loans defualt the most, then renewable energy and education

plt.figure(figsize=(16, 6))

plot_cat('purpose')
# let's also observe the distribution of loans across years

# first lets convert the year column into datetime and then extract year and month from it

df['issue_d'].head()
from datetime import datetime

df['issue_d'] = df['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))

# extracting month and year from issue_date

df['month'] = df['issue_d'].apply(lambda x: x.month)

df['year'] = df['issue_d'].apply(lambda x: x.year)





# let's first observe the number of loans granted across years

df.groupby('year').year.count()
# number of loans across months

df.groupby('month').month.count()
# lets compare the default rates across years

# the default rate had suddenly increased in 2011, inspite of reducing from 2008 till 2010

plot_cat('year')
# comparing default rates across months: not much variation across months

plt.figure(figsize=(16, 6))

plot_cat('month')
# loan amount: the median loan amount is around 10,000

sns.distplot(df['loan_amnt'])

plt.show()
# binning loan amount

def loan_amount(n):

    if n < 5000:

        return 'low'

    elif n >=5000 and n < 15000:

        return 'medium'

    elif n >= 15000 and n < 25000:

        return 'high'

    else:

        return 'very high'

        

df['loan_amnt'] = df['loan_amnt'].apply(lambda x: loan_amount(x))

df['loan_amnt'].value_counts()
# let's compare the default rates across loan amount type

# higher the loan amount, higher the default rate

plot_cat('loan_amnt')
# let's also convert funded amount invested to bins

df['funded_amnt_inv'] = df['funded_amnt_inv'].apply(lambda x: loan_amount(x))
# funded amount invested

plot_cat('funded_amnt_inv')
# lets also convert interest rate to low, medium, high

# binning loan amount

def int_rate(n):

    if n <= 10:

        return 'low'

    elif n > 10 and n <=15:

        return 'medium'

    else:

        return 'high'

    

    

df['int_rate'] = df['int_rate'].apply(lambda x: int_rate(x))
# comparing default rates across rates of interest

# high interest rates default more, as expected

plot_cat('int_rate')
# debt to income ratio

def dti(n):

    if n <= 10:

        return 'low'

    elif n > 10 and n <=20:

        return 'medium'

    else:

        return 'high'

    



df['dti'] = df['dti'].apply(lambda x: dti(x))
# comparing default rates across debt to income ratio

# high dti translates into higher default rates, as expected

plot_cat('dti')
# funded amount

def funded_amount(n):

    if n <= 5000:

        return 'low'

    elif n > 5000 and n <=15000:

        return 'medium'

    else:

        return 'high'

    

df['funded_amnt'] = df['funded_amnt'].apply(lambda x: funded_amount(x))
plot_cat('funded_amnt')

# installment

def installment(n):

    if n <= 200:

        return 'low'

    elif n > 200 and n <=400:

        return 'medium'

    elif n > 400 and n <=600:

        return 'high'

    else:

        return 'very high'

    

df['installment'] = df['installment'].apply(lambda x: installment(x))
# comparing default rates across installment

# the higher the installment amount, the higher the default rate

plot_cat('installment')
# annual income

def annual_income(n):

    if n <= 50000:

        return 'low'

    elif n > 50000 and n <=100000:

        return 'medium'

    elif n > 100000 and n <=150000:

        return 'high'

    else:

        return 'very high'



df['annual_inc'] = df['annual_inc'].apply(lambda x: annual_income(x))
# annual income and default rate

# lower the annual income, higher the default rate

plot_cat('annual_inc')
# employment length

# first, let's drop the missing value observations in emp length

df = df[~df['emp_length'].isnull()]



# binning the variable

def emp_length(n):

    if n <= 1:

        return 'fresher'

    elif n > 1 and n <=3:

        return 'junior'

    elif n > 3 and n <=7:

        return 'senior'

    else:

        return 'expert'



df['emp_length'] = df['emp_length'].apply(lambda x: emp_length(x))
# emp_length and default rate

# not much of a predictor of default

plot_cat('emp_length')
# purpose: small business loans defualt the most, then renewable energy and education

plt.figure(figsize=(16, 6))

plot_cat('purpose')
# lets first look at the number of loans for each type (purpose) of the loan

# most loans are debt consolidation (to repay otehr debts), then credit card, major purchase etc.

plt.figure(figsize=(16, 6))

sns.countplot(x='purpose', data=df)

plt.show()
# filtering the df for the 4 types of loans mentioned above

main_purposes = ["credit_card","debt_consolidation","home_improvement","major_purchase"]

df = df[df['purpose'].isin(main_purposes)]

df['purpose'].value_counts()
# plotting number of loans by purpose 

sns.countplot(x=df['purpose'])

plt.show()
# let's now compare the default rates across two types of categorical variables

# purpose of loan (constant) and another categorical variable (which changes)



plt.figure(figsize=[10, 6])

sns.barplot(x='term', y="loan_status", hue='purpose', data=df)

plt.show()

# lets write a function which takes a categorical variable and plots the default rate

# segmented by purpose 



def plot_segmented(cat_var):

    plt.figure(figsize=(10, 6))

    sns.barplot(x=cat_var, y='loan_status', hue='purpose', data=df)

    plt.show()



    

plot_segmented('term')
# grade of loan

plot_segmented('grade')
# home ownership

plot_segmented('home_ownership')
# year

plot_segmented('year')
# emp_length

plot_segmented('emp_length')
# loan_amnt: same trend across loan purposes

plot_segmented('loan_amnt')
# interest rate

plot_segmented('int_rate')
# installment

plot_segmented('installment')
# debt to income ratio

plot_segmented('dti')
# annual income

plot_segmented('annual_inc')
# variation of default rate across annual_inc

df.groupby('annual_inc').loan_status.mean().sort_values(ascending=False)
# one can write a function which takes in a categorical variable and computed the average 

# default rate across the categories

# It can also compute the 'difference between the highest and the lowest default rate' across the 

# categories, which is a decent metric indicating the effect of the varaible on default rate



def diff_rate(cat_var):

    default_rates = df.groupby(cat_var).loan_status.mean().sort_values(ascending=False)

    return (round(default_rates, 2), round(default_rates[0] - default_rates[-1], 2))



default_rates, diff = diff_rate('annual_inc')

print(default_rates) 

print(diff)

# filtering all the object type variables

df_categorical = df.loc[:, df.dtypes == object]

df_categorical['loan_status'] = df['loan_status']



# Now, for each variable, we can compute the incremental diff in default rates

print([i for i in df.columns])
# storing the diff of default rates for each column in a dict

d = {key: diff_rate(key)[1]*100 for key in df_categorical.columns if key != 'loan_status'}

print(d)