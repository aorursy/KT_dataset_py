import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pd.set_option('display.max_columns', None)
data = pd.read_csv('../input/loan.csv')

print(data.duplicated().sum())

data.shape
# see first rows

data.head()
fig = data.loan_amnt.hist(bins=50)

fig.set_title('Loan Requested Amount')

fig.set_xlabel('Loan Amount')

fig.set_ylabel('Number of loans')
# parse loan requested date as datetime, to make the below plots

data['issue_dt'] = pd.to_datetime(data.issue_d)

data['month']= data['issue_dt'].dt.month

data['year']= data['issue_dt'].dt.year

data[['issue_d', 'issue_dt', 'month', 'year']].head()
# plot total loan amount lent in time, segregated by the different risk markets (variable grade)

fig = data.groupby(['year', 'grade'])['loan_amnt'].sum().unstack().plot()

fig.set_title('Loan Requested Amount')

fig.set_ylabel('Loan Amount')

fig.set_xlabel('Time')
bad_indicators = ["Charged Off ",

                    "Default",

                    "Does not meet the credit policy. Status:Charged Off",

                    "In Grace Period", 

                    "Default Receiver", 

                    "Late (16-30 days)",

                    "Late (31-120 days)"]



# define a bad loan

data['bad_loan'] = 0

data.loc[data.loan_status.isin(bad_indicators), 'bad_loan'] = 1

data.bad_loan.mean()
dict_risk = data.groupby(['grade'])['bad_loan'].mean().sort_values().to_dict()

dict_risk
fig = data.groupby(['grade'])['bad_loan'].mean().sort_values().plot.bar()

fig.set_ylabel('Percentage of bad debt')
fig = data.groupby(['grade'])['int_rate'].mean().plot.bar()

fig.set_ylabel('Interest Rate')
fig = data.groupby(['grade'])['loan_amnt'].sum().plot.bar()

fig.set_ylabel('Loan amount disbursed')
fig = data.groupby(['grade', 'term'])['loan_amnt'].mean().unstack().plot.bar()

fig.set_ylabel('Mean loan amount disbursed')
fig = data.groupby(['grade', 'term'])['bad_loan'].mean().unstack().plot.bar()

fig.set_ylabel('Percentage of bad debt')