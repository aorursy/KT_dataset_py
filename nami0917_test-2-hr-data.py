# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



from matplotlib_venn import venn3, venn3_circles



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd.read_csv('../input/HR_comma_sep.csv')
hr = pd.read_csv('../input/HR_comma_sep.csv')

hr.shape
hr.head(10)
hr.describe()
hr.corr() # corr of all data 
hr['time_spend_company'].unique()
data0 = hr[['time_spend_company','satisfaction_level']]

data0 = data0.groupby('time_spend_company').agg(['mean']).reset_index()

data0.columns = data0.columns.get_level_values(0)

data0.plot(x='time_spend_company', y='satisfaction_level')

plt.show()
data1 = hr[['left', 'time_spend_company']]

data1 = data1.groupby('time_spend_company').agg(['sum', 'count']).reset_index()

data1.columns = data1.columns.get_level_values(0)

data1.columns = ['time_spend_company', 'left', 'total']

data1.plot(x='time_spend_company',y=['left','total'], kind='bar')

plt.show()
hr['sales'].unique()
mydims = (10, 5)

fig, ax = plt.subplots(figsize=mydims)

sns.barplot(hr['sales'], hr['satisfaction_level'], ax=ax)
data2 = hr[['left', 'sales']]

data2 = data2.groupby('sales').agg(['sum', 'count']).reset_index()

data2.columns = data2.columns.get_level_values(0)

data2.columns = ['Department', 'left', 'total']

data2.plot(x='Department',y=['left', 'total'],kind='bar')

plt.show()
hr['salary'].unique()
corrcol_low = hr[hr['salary'] == 'low'].corr()

corrcol_medium = hr[hr['salary'] == 'medium'].corr()

corrcol_high = hr[hr['salary'] == 'high'].corr()



sns.heatmap(corrcol_low, vmax=.8, square=True,annot=True,fmt='.2f'),
corrcol_low = hr[hr['salary'] == 'low'].corr()

corrcol_medium = hr[hr['salary'] == 'medium'].corr()

corrcol_high = hr[hr['salary'] == 'high'].corr()



sns.heatmap(corrcol_medium, vmax=.8, square=True,annot=True,fmt='.2f')
corrcol_low = hr[hr['salary'] == 'low'].corr()

corrcol_medium = hr[hr['salary'] == 'medium'].corr()

corrcol_high = hr[hr['salary'] == 'high'].corr()



sns.heatmap(corrcol_high, vmax=.8, square=True,annot=True,fmt='.2f')
data4 = hr[['time_spend_company', 'salary','left']]

data4 = data4.groupby(['time_spend_company','salary']).agg(['sum']).reset_index()

data4.columns = data4.columns.get_level_values(0)

# high=>3, medium=>2, low=>1

data4.loc[:,'salary'] = data4.apply(lambda row : {'high':3, 'medium':2, 'low':1}[row['salary']], 1)

data4.plot(x='time_spend_company', y='salary', s=data4['left'], kind='scatter')

plt.show()
hr.groupby('left').count()['salary'].plot(kind='bar', color="Blue", title='Stayed "0" VS. Left "1"', width =.3,stacked = True)

print('The # of Employees Left = {} \n Total # of Employees = {}'.format(hr[hr['left']==1].shape[0], hr.shape[0]))

order = ['low', 'medium', 'high']

hr.groupby('salary').count()['sales'].loc[order].plot(kind='bar', color='rgb', title='The Percentile of each Salary Class', width =.3,stacked = True)



print('The # of Employees with low salary = {} \n The # of Employees with medium salary = {} \n The # of Employees with high salary = {}'.format(hr[hr['salary']=='low'].shape[0], hr[hr['salary']=='medium'].shape[0], hr[hr['salary']=='high'].shape[0]))
corrcol2 = hr.corr()

f, ax =plt.subplots(figsize=(10,5))

sns.heatmap(corrcol2, vmax=.8, square=True,annot=True,fmt='.2f')

plt.title('Correlation between variables')
left = hr[(hr['left'] == 1)]

stay = hr[(hr['left']==0)]

department_name = hr['sales'].unique()

name=['Sales','Accounting','HR','Technical','Support','Management','IT','Product Management','Marketing','R&D']

index = range(10)



plt.figure(1,figsize=(12,6))

data5 = hr[['left', 'sales']]

data5 = data5.groupby('sales').agg(['sum', 'count']).reset_index()

data5.columns = data5.columns.get_level_values(0)

data5.columns = ['department_name', 'left', 'Stay']

data5.plot(x='department_name',y=['left', 'Stay'],kind='bar')

plt.show()
L_salary_level_count = left['salary'].value_counts()

S_salary_level_count = stay['salary'].value_counts()



plt.figure(1,figsize=(10,5))

plt.subplot(1,2,1)

L_salary_level_count.plot(kind='bar',rot=0)

plt.title('The number of employees who \n LEAVE the company by salary level ')



plt.subplot(1,2,2)

S_salary_level_count.plot(kind='bar',rot=0,color='green')

plt.title('The number of employees who \n STAY the company by salary level ')



hr['Current_Status'] = hr['left'].apply(lambda x: 'Stay' if x == 0 else 'Left')

hr.groupby(['sales','salary','Current_Status']).size()
def plot_department_left_salary(department):

    department_left = left[left['sales'] ==department]

    count = department_left['salary'].value_counts()

    index = [1,2,3]

#     color = ['red','blue','green']

    plt.bar(index,count,width=0.5)

    plt.xticks(index,['Low','Medium','High'])

    

def plot_department_stay_salary(department):

    department_stay = stay[stay['sales'] ==department]

    count = department_stay['salary'].value_counts()

    index = [1,2,3]

    color = ['rgb']

    plt.bar(index,count,width=0.5,color='green')

    plt.xticks(index,['Low','Medium','High'])

    

plt.figure(1,figsize=(15,10))

for i in range(10):

    plt.subplot(2,5,i+1)

    plot_department_left_salary(department_name[i])

    plt.title(name[i])

plt.suptitle('Employees who left by salary classes from each department')   

    

    