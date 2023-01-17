import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
use_cols = ['id', 'purpose', 'loan_status', 'home_ownership']
data = pd.read_csv(
    '../input/lending-club-loan-data/loan.csv', usecols=use_cols).sample(10000, random_state=44)
data.head()
fig=data['home_ownership'].value_counts().plot.bar()
fig = data['purpose'].value_counts().plot.bar()
fig.set_title('Loan Purpose')
fig.set_ylabel('Number of customers')

fig = data['loan_status'].value_counts().plot.bar()
fig.set_title('Status of the Loan')
fig.set_ylabel('Number of customers')
