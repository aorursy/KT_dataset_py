# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import seaborn as sns
sns.set(color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sal_all = pd.read_table('../input/Salaries.csv', sep=r',', skipinitialspace=True)
sal_all.head()
sal_all.shape
sal_all.head()


sal_info = sal_all[sal_all.TotalPay>1000]
for col in ['BasePay','OvertimePay','OtherPay', 'Benefits']:
    sal_info[col] = pd.to_numeric(sal_info[col], errors='coerce')

sal_info=sal_info[sal_info.BasePay.notnull()]
sal_info=sal_info[sal_info.TotalPay.notnull()]
sal_info=sal_info[sal_info.TotalPayBenefits.notnull()]
sal_info.loc[:,'EmployeeName'] = sal_info.loc[:,'EmployeeName'].str.upper().str.replace('  ',' ')
#sal_info.loc[:,'EmployeeName'] = sal_info.loc[:,'EmployeeName']
sal_info.loc[:,'JobTitle'] = sal_info.loc[:,'JobTitle'].str.upper().str.replace('  ',' ')
sal_info = sal_info[sal_info.EmployeeName != 'NOT PROVIDED']

s1 = sal_info.groupby(['Year','EmployeeName']).TotalPayBenefits.mean().unstack('Year').fillna(0)
s1.head()

s1 = sal_info.groupby(['Year','EmployeeName']).TotalPay.mean().unstack('Year').fillna(0)
s1.head()
s3 = sal_info.groupby(['Year','JobTitle']).TotalPay.mean().unstack('Year').fillna(0)
s3.head()
s3.loc[:,2011].sort_values(ascending=False).head(10)
s3.loc[:,2014].sort_values(ascending=False).head(10)

sal_info=sal_info.replace({'CHIEF OF DEPARTMENT, (FIRE DEPARTMENT)' : 'CHIEF, FIRE DEPARTMENT'})
sal_info=sal_info.replace({'ASSISTANT MEDICAL EXAMINER' : 'ASST MED EXAMINER'})
sal_info=sal_info.replace({'DEPUTY CHIEF III (POLICE DEPARTMENT)' : 'DEPUTY CHIEF 3'})
sal_info=sal_info.replace({'DEPARTMENT HEAD V' : 'DEPT HEAD V'})
sal_info=sal_info.replace({'DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)' : 'ASSISTANT DEPUTY CHIEF 2'})
sal_info=sal_info.replace({'DEPUTY DIRECTOR I - MUNICIPAL TRANSPORTATION AGE' : 'DEPUTY DIR I, MTA'})
sal_info=sal_info.replace({'DEPUTY DIRECTOR V' : 'DEP DIR V'})
sal_info=sal_info.replace({'LIEUTENANT, BUREAU OF FIRE PREVENTION AND PUBLIC S' : 'LIEUT,FIRE PREV'})
sal_info=sal_info.replace({'CONFIDENTIAL CHIEF ATTORNEY II (CIVIL & CRIMINAL)' : 'CFDNTAL CHF ATTY 2,(CVL&CRMNL)'})
sal_info=sal_info.replace({'CAPTAIN III (POLICE DEPARTMENT)' : 'CAPTAIN 3'})
sal_info=sal_info.replace({'DEPUTY DIRECTOR II - MUNICIPAL TRANSPORTATION AG' : 'DEPUTY DIR II, MTA'})
sal_info=sal_info.replace({'BATTALION CHIEF, (FIRE DEPARTMENT)' : 'BATTALION CHIEF, FIRE SUPPRESS'})
sal_info=sal_info.replace({'DEPARTMENT HEAD IV' : 'DEPT HEAD IV'})
sal_info=sal_info.replace({'COMMANDER III, (POLICE DEPARTMENT)' : 'CAPTAIN 3'})
sal_info=sal_info.replace({'ADMINISTRATOR, SFGH MEDICAL CENTER' : 'ADM, SFGH MEDICAL CENTER'})
sal_info=sal_info.replace({'GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY' : 'GEN MGR, PUBLIC TRNSP DEPT'})
sal_info=sal_info.replace({'ASSISTANT DEPUTY CHIEF II' : 'ASSISTANT DEPUTY CHIEF 2'})
sal_info=sal_info.replace({'DEPUTY DIRECTOR OF INVESTMENTS' : 'DEP DIR FOR INVESTMENTS, RET'})
sal_info=sal_info.replace({'BATTLION CHIEF, FIRE SUPPRESSI' : 'BATTALION CHIEF, FIRE SUPPRESS'})
sal_info=sal_info.replace({'BATTLION CHIEF, FIRE SUPPRESS' : 'BATTALION CHIEF, FIRE SUPPRESS'})
sal_info=sal_info.replace({'ASSISTANT CHIEF OF DEPARTMENT, (FIRE DEPARTMENT)' : 'ASST CHF OF DEPT (FIRE DEPT)'})
s3 = sal_info.groupby(['Year','JobTitle']).TotalPay.mean().unstack('Year').fillna(0)
s3.head()
s3.loc[:,2011].sort_values(ascending=False).head(20)


s3.loc[:,2014].sort_values(ascending=False).head(20)
s3.sort_values(by=[(2011)], ascending=False).head(10).plot(kind='barh')
plt.title('Top 10 Pay - Job Titles')
s3.sort_values(by=[(2014)], ascending=False).head(10).plot(kind='barh')
plt.title('Top 10 Pay - Job Titles')
sal_info.groupby('Year')['BasePay','TotalPay','TotalPayBenefits'].mean().plot(kind='bar')
plt.title('Distribution of Average Pay across all SF City Employess')
sal_info.groupby('Year')['BasePay','OtherPay','Benefits'].mean().plot(kind='bar',stacked=True)
top_ten_positions = sal_info.JobTitle.value_counts().sort_values(ascending=False).head(10).index
sal_info[sal_info.JobTitle.isin(top_ten_positions)].groupby(['Year','JobTitle'])['TotalPay'].mean().unstack('Year')
# Seems like PATIENT CARE ASSISTANT, PUBLIC SVC AIDE-PUBLIC WORKS and POLICE OFFICER 3 had different title names in 2011

top_ten_positions = sal_info.JobTitle.value_counts().sort_values(ascending=False).head(10).index
sal_info[sal_info.JobTitle.isin(top_ten_positions)].groupby(['Year','JobTitle'])['TotalPay'].mean().unstack('Year')
sal_info[sal_info.JobTitle.isin(top_ten_positions)].groupby(['Year','JobTitle'])['TotalPay'].mean().unstack('Year').plot(kind='barh')
plt.title('Total pay for top 10 SF City positions')
s4 = sal_info.groupby(['Year','JobTitle']).TotalPay.mean().unstack('Year').fillna(0)
sns.kdeplot(s4.loc[s4.loc[:,2011]>100,2011], shade=True, cut=0)
sns.kdeplot(s4.loc[s4.loc[:,2012]>100,2012], shade=True, cut=0)
sns.kdeplot(s4.loc[s4.loc[:,2013]>100,2013], shade=True, cut=0)
sns.kdeplot(s4.loc[s4.loc[:,2014]>100,2014], shade=True, cut=0)
plt.title('Total Pay distribution over years')






