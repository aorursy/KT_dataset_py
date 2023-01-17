%matplotlib inline
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import pandas as pd
loans = pd.read_csv("../input/loan.csv", index_col = 0, low_memory=False)

Tot_loans = loans.loan_amnt.value_counts()

#Tot_loans.plot(kind = 'bar',figsize=(250,10), title = 'Loan Amounts')

y = loans.loan_amnt.values
y.plot(kind = 'bar',x = 'Amount')
#loans.drop('int_rate',axis = 1, inplace=True)
# This is ALOTTTTTT of data. Will need to clean this up later... Looks like 60k
# were loaned about 10k dollars. Need to fix plot, too big.
loan_location = loans.zip_code.value_counts()

loan_location.plot(kind = 'bar',figsize=(250,10), title = 'Loans per zipcode')

# So I am trying to understand where are the borrowers live. Looks like Cali.
term = loans.term.value_counts()

term.plot(kind = 'bar',figsize=(16,8), title = 'Term of Loan')
# The average borrower is almost 3x more likely to have a 36 month loan compared to a 60 month loan.

y = loans.int_rate.values
loans.drop('int_rate',axis = 1, inplace=True)

np.unique(y), pd.Series(y).plot(kind='hist',alpha=.7, bins=20, title='Interest Rate Distribution')

# What the distribution of interest rates are given the 
