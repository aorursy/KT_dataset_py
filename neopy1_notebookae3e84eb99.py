# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv', low_memory=False)

df = df[df['Employment Type'].isin(['Full Time', 'Part Time'])]
for i in ['Projected Annual Salary','Q1 Payments','Q2 Payments','Q3 Payments','Q4 Payments','Payments Over Base Pay','Total Payments','Base Pay','Permanent Bonus Pay','Longevity Bonus Pay','Temporary Bonus Pay','Overtime Pay','Other Pay & Adjustments','Other Pay (Payroll Explorer)','Average Health Cost','Average Dental Cost','Average Basic Life','Average Benefit Cost']:

    df[i] = df[i].str.replace('$','')
for i in ['Projected Annual Salary','Q1 Payments','Q2 Payments','Q3 Payments','Q4 Payments','Payments Over Base Pay','Total Payments','Base Pay','Permanent Bonus Pay','Longevity Bonus Pay','Temporary Bonus Pay','Overtime Pay','Other Pay & Adjustments','Other Pay (Payroll Explorer)','Average Health Cost','Average Dental Cost','Average Basic Life','Average Benefit Cost']:

    df[i] = pd.to_numeric(df[i])
df.info()
df.groupby('Department Title')['Projected Annual Salary'].mean().plot(kind='bar', figsize=(20,5), title='Avg salary across depts.')
#diff in max n min salaries for the dept across years

df_byyear = df.pivot_table(index='Year', columns='Department Title', values='Projected Annual Salary', aggfunc=lambda x: x.max()-x.min())

df_byyear
df_byyear_max = df.pivot_table(index='Year', columns='Department Title', values='Projected Annual Salary', aggfunc=lambda x: x.max())

df_byyear_min = df.pivot_table(index='Year', columns='Department Title', values='Projected Annual Salary', aggfunc=lambda x: x.min())

df_byyear_min

#Min salary by dept and year
#Max salary by dept and year

df_byyear_max
#employee counts by dept and year

df_emp_count = df.groupby('Year')['Department Title'].value_counts()

df_emp_count = pd.DataFrame(df_emp_count).unstack()

#employee count dataframe

df_emp_count
#dept with the max employee count year-wise

for i in df_emp_count.index:

    print (df_emp_count.ix[i, :].idxmax(), df_emp_count.ix[i, :].max())
import matplotlib.plot as plt

import matplotlib.gridspec as gdspc

gs = gdspc.Gridspec(5,5)

ax1=plt.subplot(gs[0])

ax2=plt.subplot(gs[1])

ax3=plt.subplot(gs[2])

ax4=plt.subplot(gs[3])

df_emp_count.T.plot(kind='bar', subplots=True, ax=ax1)