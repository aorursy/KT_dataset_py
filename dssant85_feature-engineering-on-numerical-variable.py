import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# let's load the dataset with just a few columns and a few rows
# to speed up  things

use_cols = [
    'loan_amnt', 'int_rate', 'annual_inc', 'open_acc', 'loan_status',
    'open_il_12m'
]

data = pd.read_csv(
    '../input/lending-club-loan-data/loan.csv', usecols=use_cols).sample(
        10000, random_state=44)  # set a seed for reproducibility

data.head()
# let's look at the values of the variable loan_amnt
# this is the amount of money requested by the borrower
# in US dollars

data.loan_amnt.unique()
 # let's make an histogram to get familiar with the
# distribution of the variable

fig = data.loan_amnt.hist(bins=50)
fig.set_title('Loan Amount Requested')
fig.set_xlabel('Loan Amount')
fig.set_ylabel('Number of Loans')
# let's do the same exercise for the variable interest rate,
# which is charged by lending club to the borrowers

data.int_rate.unique()
# let's make an histogram to get familiar with the
# distribution of the variable

fig = data.int_rate.hist(bins=30)
fig.set_title('Interest Rate')
fig.set_xlabel('Interest Rate')
fig.set_ylabel('Number of Loans')
# and now,let's explore the income declared by the customers,
# that is, how much they earn yearly.

fig = data.annual_inc.hist(bins=100)
fig.set_xlim(0, 400000)
fig.set_title("Customer's Annual Income")
fig.set_xlabel('Annual Income')
fig.set_ylabel('Number of Customers')
# let's inspect the values of the variable

data.open_acc.dropna().unique()
# let's make an histogram to get familiar with the
# distribution of the variable

fig = data.open_acc.hist(bins=100)
fig.set_xlim(0, 30)
fig.set_title('Number of open accounts')
fig.set_xlabel('Number of open accounts')
fig.set_ylabel('Number of Customers')
# let's inspect the variable values

data.open_il_12m.unique()
# let's make an histogram to get familiar with the
# distribution of the variable

fig = data.open_il_12m.hist(bins=50)
fig.set_title('Number of installment accounts opened in past 12 months')
fig.set_xlabel('Number of installment accounts opened in past 12 months')
fig.set_ylabel('Number of Borrowers')
# let's inspect the values of the variable loan status

data.loan_status.unique()
# let's create one additional variable called defaulted.
# This variable indicates if the loan has defaulted, which means,
# if the borrower failed to re-pay the loan, and the money
# is deemed lost.

data['defaulted'] = np.where(data.loan_status.isin(['Default']), 1, 0)
data.defaulted.mean()
# the new variable takes the value of 0
# if the loan is not defaulted

data.head(10)
# the new variable takes the value 1 for loans that
# are defaulted

data[data.loan_status.isin(['Default'])].head()
# A binary variable, can take 2 values. For example,
# the variable defaulted that we just created:
# either the loan is defaulted (1) or not (0)

data.defaulted.unique()
# let's make a histogram, although histograms for
# binary variables do not make a lot of sense

fig = data.defaulted.hist()
fig.set_xlim(0, 2)
fig.set_title('Defaulted accounts')
fig.set_xlabel('Defaulted')
fig.set_ylabel('Number of Loans')