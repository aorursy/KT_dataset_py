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
df = pd.read_csv('../input/HR_comma_sep.csv')

df.head()
df['sales'].value_counts()
df['salary'].value_counts()
df.ix[0]
satisf_by_sales_salary = df.pivot_table('satisfaction_level', index = 'sales', columns = 'salary', aggfunc='mean')

satisf_by_sales_salary
n_sales = df.groupby('sales').size()

n_sales
n_sales_1000 = n_sales.index[n_sales >= 1000]

n_sales_1000
satisf_by_sales_salary = satisf_by_sales_salary.ix[n_sales_1000]

satisf_by_sales_salary
satisf_by_sales_salary.sort_values(by = 'high', ascending = False)
satisf_by_sales_salary['dif_sal'] = satisf_by_sales_salary['high'] - satisf_by_sales_salary['low']

satisf_by_sales_salary
satisf_by_sales_salary = satisf_by_sales_salary.sort_values(by = 'dif_sal', ascending = False)

satisf_by_sales_salary
satisf_by_sales_salary[::-1]
df.groupby('left')['satisfaction_level'].std()

#Is the same as using df.groupby('left').satisfaction_level.std()
df['left'].value_counts()