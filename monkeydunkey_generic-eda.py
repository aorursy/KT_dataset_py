# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
datasets = {}
datadir = '../input'
for f in os.listdir(datadir):
    datasets[f.replace('.csv', '')] = pd.read_csv(os.path.join(datadir, f))
datasets.keys()
datasets['kiva_loans'].head()
def barplots(df, x, y, fig_width = 16, fig_height= 6, rot = 45, ax = None):
    plt.figure(figsize = (fig_width, fig_height))
    plt.xticks(rotation=rot)
    if type(df) == type('str'):
        return sns.barplot(x = x, y =y, ax = ax, data = datasets[df].groupby(x)[y].sum().sort_values().reset_index())
    else:
        return sns.barplot(x = x, y =y, ax = ax, data = df.groupby(x)[y].sum().sort_values().reset_index())

x, y = 'sector', 'loan_amount'
barplots('kiva_loans', x, y)
x, y = 'repayment_interval', 'loan_amount'
barplots('kiva_loans', x, y)
def getLoanAmtGender(loan_row):
    loan = float(loan_row.loan_amount)

    borrowers = loan_row.borrower_genders.strip(',').split(', ')
    male_borrow_amt = loan * len(list(filter(lambda x: x == 'male', borrowers))) / len(borrowers)
    female_borrow_amt = loan * len(list(filter(lambda x: x == 'female', borrowers))) / len(borrowers)
    loan_row['male_borrow_amt'] = male_borrow_amt
    loan_row['female_borrow_amt'] = female_borrow_amt
    return loan_row
loanAmtGender = datasets['kiva_loans'][~datasets['kiva_loans'].borrower_genders.isnull()].apply(getLoanAmtGender, axis = 1)
loanAmtGender
x, y = 'male_borrow_amt', 'female_borrow_amt'
sns.barplot(x = 'borrowed by', y ='amt borrowed', data = loanAmtGender[[x, y]].sum().reset_index().rename(columns={'index': 'borrowed by', 0: 'amt borrowed'}))
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111)
loanAmtGender.groupby('repayment_interval')['female_borrow_amt'].sum().plot.bar(ax=ax, width=0.2, position=1, color=sns.color_palette("Paired", n_colors=1))
loanAmtGender.groupby('repayment_interval')['male_borrow_amt'].sum().plot.bar(ax=ax, width=0.2, position=0, color=sns.color_palette("Paired", n_colors=4)[-2])
ax.legend(['Females', 'Males'])
fig = plt.figure(figsize = (16, 6)) # Create matplotlib figure
ax = fig.add_subplot(111)
loanAmtGender.groupby('sector')['female_borrow_amt'].sum().plot.bar(ax=ax, width=0.2, position=1, color=sns.color_palette("Paired", n_colors=1))
loanAmtGender.groupby('sector')['male_borrow_amt'].sum().plot.bar(ax=ax, width=0.2, position=0, color=sns.color_palette("Paired", n_colors=4)[-2])
ax.legend(['Females', 'Males'])
x, y = 'country', 'loan_amount'
barplots('kiva_loans', x, y, fig_width = 24, fig_height= 8, rot = 90)
minmumLoanCutOff = 100000
plt.figure(figsize = (16, 6))
plt.xticks(rotation=45)
sns.barplot(x = 'sector', y ='loan_amount', data = datasets['kiva_loans'][datasets['kiva_loans'].country == 'Philippines'].groupby('sector').loan_amount.sum().reset_index())
f, axes = plt.subplots(3, 1, figsize = (16, 20))
countries = ['Kenya', 'United States', 'Peru']
for i, country in enumerate(countries):
        sns.barplot(x = 'sector', y ='loan_amount', data = datasets['kiva_loans'][datasets['kiva_loans'].country == country].groupby('sector').loan_amount.sum().reset_index(), ax=axes[i])
        axes[i].set_title(country)