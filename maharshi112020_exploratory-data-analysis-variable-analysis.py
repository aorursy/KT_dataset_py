import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_rows',None)

pd.set_option('display.max_columns', None)
df = pd.read_csv('../input/lending-club-loan-data/loan.csv')

df.head()
use_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'open_acc', 'loan_status', 'open_il_12m']

data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=use_cols).sample(10000, random_state=44)   # set a seed for reproducibility

data.sample(5)
data.shape
data.loan_amnt.unique()
fig = data.int_rate.hist(bins=50)

fig.set_title("Interest Rate Graph")
fig = data.loan_amnt.hist(bins=50)

fig.set_title("Loan Amount Distributed")

fig.set_xlabel("Loan Amount")

fig.set_ylabel("Number of Loans")
fig = data.int_rate.hist(bins=50)

fig.set_title("Interest Rate Distribution")

fig.set_xlabel("Interest Rate")

fig.set_ylabel("Number of Loans")
fig = data.annual_inc.hist(bins=100)

# fig.set_xlim(0, 400000)

fig.set_title("Customer's Annual Income")

fig.set_xlabel('Annual Income')

fig.set_ylabel('Number of Customers')
data.open_acc.unique()
data.open_acc.dropna().unique()
fig = data.open_acc.hist(bins=100)

fig.set_xlim(0, 30)

fig.set_title('Number of open accounts')

fig.set_xlabel('Number of open accounts')

fig.set_ylabel('Number of Customers')
data.open_il_12m.unique()
fig = data.open_il_12m.hist(bins=50)

fig.set_title('Number of installment accounts opened in past 12 months')

fig.set_xlabel('Number of installment accounts opened in past 12 months')

fig.set_ylabel('Number of Borrowers')
data.loan_status.unique()
data['defaulted'] = np.where(data.loan_status.isin(['Default']), 1, 0)
data.sample(5)
data['Charged off'] = np.where(data.loan_status.isin(['Charged off']), 1, 0)

data.sample(5)
fig = data.defaulted.hist()

fig.set_xlim(0, 2)

fig.set_title('Defaulted accounts')

fig.set_xlabel('Defaulted')

fig.set_ylabel('Number of Loans')
use_cols = ['id', 'purpose', 'loan_status', 'home_ownership']

data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=use_cols).sample(10000, random_state=44)

data.sample(5)
data.home_ownership.unique()
fig = data['home_ownership'].value_counts().plot.bar()

fig.set_title('home_ownership')

fig.set_ylabel('Number of customers')
data['home_ownership'].value_counts()
data.purpose.unique()
fig = data['purpose'].value_counts().plot.bar()

fig.set_title('Loan Purpose')

fig.set_ylabel('Number of customers')
fig = data.purpose.value_counts().plot.line()
fig = data.purpose.value_counts().plot.area()
use_cols = ['loan_amnt', 'grade', 'purpose', 'issue_d', 'last_pymnt_d']

data = pd.read_csv('../input/lending-club-loan-data/loan.csv', usecols=use_cols)

data.sample(5)
data.dtypes
data['issue_date'] = pd.to_datetime(data.issue_d)

data['last_pymnt_date'] = pd.to_datetime(data.last_pymnt_d)

data[['issue_d', 'issue_date', 'last_pymnt_d', 'last_pymnt_date']].head()
data.dtypes
fig = data.groupby(['issue_date', 'grade'])['loan_amnt'].sum().unstack().plot(figsize=(14, 8))

fig.set_title('Distributed amount with time')

fig.set_ylabel('Distributed Amount $')
fig = data.groupby(['issue_date', 'grade'])['loan_amnt'].sum().plot(figsize=(14, 8))

fig.set_title('Distributed amount with time')

fig.set_ylabel('Distributed Amount $')