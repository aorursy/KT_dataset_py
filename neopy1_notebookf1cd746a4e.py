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

for i in ['Projected Annual Salary','Q1 Payments','Q2 Payments','Q3 Payments','Q4 Payments','Payments Over Base Pay','Total Payments','Base Pay','Permanent Bonus Pay','Longevity Bonus Pay','Temporary Bonus Pay','Overtime Pay','Other Pay & Adjustments','Other Pay (Payroll Explorer)','Average Health Cost','Average Dental Cost','Average Basic Life','Average Benefit Cost']:

    df[i] = df[i].str.replace('$','')

    df[i] = pd.to_numeric(df[i])
df.info()
df['Payroll Department'].value_counts()
df.groupby(['Year', 'Benefits Plan', 'Department Title'])['Average Benefit Cost'].mean()
#df.pivot_table(index=('Year', 'Benefits Plan'), columns='Department Title', values='Average Benefit Cost')

new_df = df[['Year','Department Title', 'Benefits Plan', 'Average Benefit Cost']]
new_df.info()
new_df= new_df[new_df['Benefits Plan'].notnull()]
#df.pivot_table(index=('Year', 'Benefits Plan'), columns='Department Title', values='Average Benefit Cost')

df_new = df.pivot_table(index='Department Title', columns='Benefits Plan', values='Average Benefit Cost')

df_new