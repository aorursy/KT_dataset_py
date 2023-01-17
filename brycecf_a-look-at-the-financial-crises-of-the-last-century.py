import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
banks_df = pd.read_csv('../input/banks.csv',

                       index_col=None,

                       skipinitialspace=True,

                       dtype={'Finanical Institution Number': object,

                              'Certificate Number': object,

                              'Total Deposits': np.float64},

                       parse_dates=['Failure Date'],

                       na_values='')

banks_df.info()
banks_df.head()
banks_df['Failure Date'] = banks_df['Failure Date'].dt.to_period('M')

monthly_failures_df = banks_df.groupby(['Failure Date', 'Institution Type']).size().reset_index()

monthly_failures_df.rename(columns={0: 'Count'}, inplace=True)

monthly_failures_df['Failure Date'] = monthly_failures_df['Failure Date'].astype(str)

monthly_failures_df.head()
institution_failures_df = banks_df.groupby(['Failure Date', 'Institution Type']).size().unstack('Institution Type').reset_index()

all_years_df = institution_failures_df.copy()

all_years_df.set_index('Failure Date', inplace=True)

ax = all_years_df.plot(figsize=(20,10),

                       fontsize=16)

ax.legend(prop={'size': 16})

ax.set_xlabel('Failure Date', fontsize=16)

ax.set_ylabel('Count', fontsize=16)

ax.set_title('Financial Institution Failures from 1934-2017', fontsize=18)
sns.factorplot(x='Failure Date',

               y='Count',

               hue='Institution Type',

               data=monthly_failures_df,

               kind='bar',

               legend=True,

               size=10)
great_depression_df = institution_failures_df[institution_failures_df['Failure Date'] < pd.Period('1945-12')]

great_depression_df.set_index('Failure Date', inplace=True)

ax = great_depression_df.plot(figsize=(15,10),

                              fontsize=16)

ax.legend(prop={'size': 16})

ax.set_xlabel('Failure Date', fontsize=16)

ax.set_ylabel('Count', fontsize=16)

ax.set_title('Financial Institution Failures from 1933-1945', fontsize=18)
loan_savings_df = institution_failures_df[(institution_failures_df['Failure Date'] < pd.Period('1995-12')) &

                                          (institution_failures_df['Failure Date'] > pd.Period('1983-01'))]

loan_savings_df.set_index('Failure Date', inplace=True)

ax = loan_savings_df.plot(figsize=(15,10),

                           fontsize=16)

ax.legend(prop={'size': 16})

ax.set_xlabel('Failure Date', fontsize=16)

ax.set_ylabel('Count', fontsize=16)

ax.set_title('Financial Institution Failures from 1933-1945', fontsize=18)