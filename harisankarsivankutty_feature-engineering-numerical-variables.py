import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
use_cols = [
    'loan_amnt', 'int_rate', 'annual_inc', 'open_acc', 'loan_status',
    'open_il_12m'
]
data = pd.read_csv(
    '../input/lending-club-loan-data/loan.csv', usecols=use_cols).sample(10000, random_state=44)
data.head()
fig = data.loan_amnt.hist(bins=50)
fig.set_title('Loan Amount Requested')
fig.set_xlabel('Loan Amount')
fig.set_ylabel('Number of loans')
fig = data.int_rate.hist(bins=50)
fig.set_title('Interest Rate')
fig.set_xlabel('Interest Rate')
fig.set_ylabel('Number of loans')
fig = data.annual_inc.hist(bins=100)
fig.set_xlim(0,400000)
fig.set_title('Annual Income')
fig.set_xlabel('Income')
fig.set_ylabel('Number of Customers')
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
data['defaulted'] = np.where(data.loan_status.isin(['Charged Off']),0,1)
data.defaulted.mean()
data[data.loan_status.isin(['Charged Off'])].head()
fig = data.defaulted.hist()
fig.set_xlim(0, 2)
fig.set_title('Defaulted accounts')
fig.set_xlabel('Defaulted')
fig.set_ylabel('Number of Loans')
use_cols1 = ['id', 'purpose', 'loan_status', 'home_ownership']
data1 = pd.read_csv(
    '../input/lending-club-loan-data/loan.csv', usecols=use_cols1).sample(
        10000, random_state=44)  # set a seed for reproducibility
data1.head()
