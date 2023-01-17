# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/loan.csv')


#Plotting interest rate vs dti. Would have expected higher DTI to have higher interest rates
plt.scatter(df.int_rate, df.dti)
plt.xlim(5,30)
plt.ylim(0,80)
plt.xlabel('Interest Rate')
plt.ylabel('DTI')
#Plotting interest rate vs max balance. Higher max balances yield lower interest rates. Time to get another credit card?
plt.scatter(df.int_rate, df.max_bal_bc)
plt.xlim(5,30)
plt.ylim(0, 85000)
plt.xlabel('Interest Rate')
plt.ylabel('max_bal_bc')
#Plotting max_bal_bc vs dti
plt.scatter(df.max_bal_bc, df.dti)
plt.xlim(0, 85000)
plt.ylim(0, 80)
plt.xlabel('Maximum current balance owed on all revolving accounts')
plt.ylabel('dti')
#Total Current Balance is total amount this person (or applicant(s)) owe. Mortgage, Credit Cards, Auto etc.
#DTI vs Total Current Balance. It looks like some people with high balances have low DTI (<40) and people 
#with high DTI (40) generally have lower balances.
plt.scatter(df.dti, df.tot_cur_bal)
plt.xlim(0,80)
plt.ylim(0, 3000000)
plt.xlabel('DTI')
plt.ylabel('tot_cur_bal')
