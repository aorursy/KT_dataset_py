# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

%matplotlib inline

# Any results you write to the current directory are saved as output.
hr = pd.read_csv('../input/HR_comma_sep.csv')

hr.head()
hr.shape
hr.columns
hr.dtypes
hr.describe()
hr.salary.unique()
pd.unique(hr.time_spend_company.ravel())
pd.unique(hr.left.ravel())
hr.sales.count()
dept_strength = hr.groupby(['sales']).salary.count()

dept_strength
value_emp = hr[hr.left ==1].sales.value_counts()

value_emp
hr[hr.left ==1].sales.value_counts().plot(kind='bar')

plt.xlabel('Department')

plt.ylabel('Count of Employees')

plt.title('Valuable Employees - department-wise')
left_emp = hr[hr.left ==0].sales.value_counts()

left_emp
left_emp = hr[hr.left ==0].sales.value_counts().plot(kind='bar')

plt.xlabel('Department')

plt.ylabel('Count of Employees')

plt.title('Employees who left the company - department-wise')
a = hr.left.value_counts()

a
a.plot('bar', rot = 0)

plt.xlabel('Employee with the company - Stay (1) or Left (0)')

plt.ylabel('Count of Employees')
s = hr.salary.value_counts()

s
hr.salary.value_counts().plot(kind='bar', rot =0)

plt.xlabel('Salary level')

plt.ylabel('Count of Employees')

plt.title('Salary of Employees')
hr[hr['salary'] == 'high'].sales.value_counts()
hr[hr['salary'] == 'high'].sales.count()
hr[hr['salary'] == 'low'].sales.count()
hr[hr['salary'] == 'medium'].sales.count()
hr[hr['satisfaction_level'] > .7].sales.value_counts()
hr[hr['satisfaction_level'] > .7].sales.count()
hr[hr['last_evaluation'] > 0.7].sales.value_counts()
hr[hr['last_evaluation'] > 0.7].sales.count()
hr[hr['last_evaluation']== 1.0].sales.value_counts()
hr[hr['last_evaluation']== 1.0].sales.count()
hr.promotion_last_5years.value_counts()
#High salaried, promotion_last_5years and last evaulation =1, and left 
x = hr[hr['left'] == 1].salary.value_counts()

x
x.plot(kind='bar')

plt.xlabel('Salary level')

plt.ylabel('Count of Employees staying with the company')
y = hr[hr['left'] == 0].salary.value_counts()

y
y.plot(kind='bar');

plt.xlabel('Salary level')

plt.ylabel('Count of Employees left the company')
hr[hr['time_spend_company'] >= 5].salary.value_counts()
hr.groupby(['sales','time_spend_company']).salary.value_counts()
zh = hr[(hr['salary'] == 'high') & (hr['time_spend_company'] >= 5)]

zh[:5]
zh['sales'].value_counts()
zh['sales'].count()
zh['sales'].value_counts().plot(kind='bar')
zl = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] >= 5)]

zl[:5]
zl['sales'].value_counts()
zl['sales'].count()
zl['sales'].value_counts().plot(kind='bar')
zh1 = hr[(hr['salary'] == 'high') & (hr['time_spend_company'] <= 5)]

zh1[:5]
zh1['sales'].value_counts()
zh1['sales'].count()
zl1 = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] <= 5)]

zl1[:5]
zl1['sales'].value_counts()
zl1['sales'].count()
hr.time_spend_company.value_counts()
hr.time_spend_company.value_counts().plot(kind='barh')

plt.ylabel('Count of years')

plt.xlabel('Count of Employees')
hr.number_project.value_counts()
hr[hr.salary == 'low'][hr.number_project >= 4]
hr.average_montly_hours.describe()
hr[hr.salary == 'low'][hr.average_montly_hours >= 245]
mz = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] <= 5) & (hr['average_montly_hours'] >= 245) & (hr['left'] >= 0) &(hr['promotion_last_5years'] >= 0)]

mz.count()
promotion_last_5years

left
hr.promotion_last_5years.value_counts()
mz1 = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] <= 6)  & (hr['left'] >= 0) &(hr['promotion_last_5years'] >= 0)]

mz1.count()
mz11 = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] >= 5)  & (hr['left'] >= 0) &(hr['promotion_last_5years'] >= 0)]

mz11.count()
mz112 = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] >= 5)  & (hr['left'] >= 0) &(hr['promotion_last_5years'] >= 1)]

mz112.count()
mz113 = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] <= 5)  & (hr['left'] >= 0) &(hr['promotion_last_5years'] >= 1)]

mz113.count()
mz12 = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] >= 5)  & (hr['left'] >= 1) &(hr['promotion_last_5years'] >= 0)]

mz12.count()
mz123 = hr[(hr['salary'] == 'low') & (hr['time_spend_company'] <= 5)  & (hr['left'] >= 1) &(hr['promotion_last_5years'] >= 0)]

mz123.count()
mz111 = hr[(hr['salary'] == 'high') & (hr['time_spend_company'] >= 5)  & (hr['left'] >= 0) &(hr['promotion_last_5years'] >= 1)]

mz111.count()
mz1111 = hr[(hr['salary'] == 'high') & (hr['time_spend_company'] <= 5)  & (hr['left'] >= 1) &(hr['promotion_last_5years'] >= 0)]

mz1111.count()
mz1111 = hr[(hr['salary'] == 'high') & (hr['time_spend_company'] >= 1)  & (hr['left'] >= 1) &(hr['promotion_last_5years'] >= 5)]

mz1111.count()