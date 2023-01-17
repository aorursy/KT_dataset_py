import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

%matplotlib inline

sns.set_style("dark")

import warnings 

warnings.filterwarnings("ignore")
loan = pd.read_csv('/kaggle/input/loandata/Loan payments data.csv')
loan.head(5)
loan.info()
loan.loan_status.unique()
fig = plt.figure(figsize=(5,5))

ax = sns.countplot(loan.loan_status)

ax.set_title("Count of Loan Status")

for p in ax.patches:

    ax.annotate(str(format(int(p.get_height()), 'd')), (p.get_x(), p.get_height()*1.01))

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(16, 5))

sns.boxplot(x="loan_status", y="Principal", data=loan, hue="loan_status", ax=axs[0])

sns.distplot(loan.Principal, bins=range(300, 1000, 100), ax=axs[1], kde=True)

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(16,5))

sns.countplot(loan.terms, ax=axs[0])

axs[0].set_title("Count of Terms of loan")

for p in axs[0].patches:

    axs[0].annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))



sns.countplot(x='terms', hue='loan_status', data=loan, ax=axs[1])

axs[1].set_title("Term count breakdown by loan_status")

for t in axs[1].patches:

    if(np.isnan(float(t.get_height()))):

        axs[1].annotate(0, (t.get_x(), 0))

    else:

        axs[1].annotate(str(format(int(t.get_height()), 'd')), (t.get_x(), t.get_height()*1.01))



axs[1].legend(loc="upper left")

plt.show()
fig = plt.figure(figsize=(10,5))

ax = sns.countplot(x='effective_date', hue='loan_status', data=loan)

ax.set_title('Loan Date')

for t in ax.patches:

    if(np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

plt.show()
loan['paid_off_date'] = pd.DatetimeIndex(loan.paid_off_time).normalize()

fig = plt.figure(figsize=(16, 6))

ax = sns.countplot(x='paid_off_date', data=loan.loc[loan.loan_status.isin(['COLLECTION_PAIDOFF', 'PAIDOFF'])], hue='loan_status')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

for t in ax.patches:

    if(np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), 'd')), (t.get_x(), t.get_height()*1.01))



ax.legend(loc='upper right')

plt.show()
loan['day_to_pay'] = (pd.DatetimeIndex(loan.paid_off_time).normalize() - pd.DatetimeIndex(loan.effective_date).normalize()) / np.timedelta64(1, 'D')



fig = plt.figure(figsize=(15,5))

ax = sns.countplot(x='day_to_pay', hue='terms', data=loan)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

for t in ax.patches:

    if(np.isnan(float(t.get_height()))):

        ax.annotate('', (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), 'd')), (t.get_x(), t.get_height()*1.01))



plt.show()
fig = plt.figure(figsize=(15, 5))

ax = sns.countplot(x='day_to_pay', hue='terms', data=loan.loc[loan.loan_status == 'PAIDOFF'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

for t in ax.patches:

    if(np.isnan(float(t.get_height()))):

        ax.annotate('', (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

        

plt.show()
fig, axs = plt.subplots(3, 2, figsize=(16,15))

sns.distplot(loan.age, ax=axs[0][0])

axs[0][0].set_title("Total age distribution across dataset")



sns.boxplot(x='loan_status', y='age', data=loan, ax=axs[0][1])

axs[0][1].set_title("Age distribution by loan status")



sns.countplot(x='education', data=loan, ax=axs[1][0])

axs[1][0].set_title("Education count")

for t in axs[1][0].patches:

    if(np.isnan(float(t.get_height()))):

        axs[1][0].annotate('', (t.get_x(), 0))

    else:

        axs[1][0].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

        

sns.countplot(x='education', data=loan, hue='loan_status', ax=axs[1][1])

axs[1][1].set_title("Education by loan status")

for t in axs[1][1].patches:

    if(np.isnan(float(t.get_height()))):

        axs[1][1].annotate('', (t.get_x(), 0))

    else:

        axs[1][1].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

axs[1][1].legend(loc='upper right')



sns.countplot(x='Gender', data=loan, ax=axs[2][0])

axs[2][0].set_title("Education of Gender")

for t in axs[2][0].patches:

    if(np.isnan(float(t.get_height()))):

        axs[2][0].annotate('', (t.get_x(), 0))

    else:

        axs[2][0].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

 

sns.countplot(x='Gender', data=loan, hue='education', ax=axs[2][1])

axs[2][1].set_title("Education of Gender")

for t in axs[2][1].patches:

    if(np.isnan(float(t.get_height()))):

        axs[2][1].annotate('', (t.get_x(), 0))

    else:

        axs[2][1].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

 

plt.show()