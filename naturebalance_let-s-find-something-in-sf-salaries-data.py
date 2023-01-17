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
temp=pd.read_csv('../input/Salaries.csv')

temp.head()
temp.describe()
temp.dtypes
temp.Year.unique()
temp.Status.unique()
temp['Id'].count()
temp.groupby('Status').count()

tempFT=temp[temp['Status']=='FT']

tempPT=temp[temp['Status']=='PT']

temp0=temp[~temp['Status'].isin(['PT','FT'])]
print('PT',tempPT['TotalPay'].median(),',','FT',tempFT['TotalPay'].median(),',','unknown',temp0['TotalPay'].median())
tempa=temp.groupby('JobTitle').count()

tempb=tempa.loc[:,['Id']]

tempc=tempb.sort_values(by='Id')
tempc.head()
tempc.tail()
tempa=temp.loc[:,['JobTitle','TotalPay']].groupby('JobTitle').median()
tempb=tempa.sort_values(by='TotalPay')
tempb.median()
tempa=temp.loc[:,['JobTitle','TotalPay']]
tempa[tempa['JobTitle']=='Police Officer'].mean()
tempa=temp[temp['JobTitle'].str.startswith('Pol')]

tempa['TotalPay'].median()
tempa['JobTitle'].unique()
tempb=tempa.loc[:,['JobTitle','TotalPay']]

tempb.groupby('JobTitle').median()
tempb.groupby('JobTitle').count()