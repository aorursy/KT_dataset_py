# Import necessary library files

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



#Read Data

data = pd.read_csv("../input/HR_comma_sep.csv")

data[0:5]
#Find the missing values

data.isnull().any()
#Data types of each feature in the dataset

data.dtypes   
data.describe()
data.groupby('left').satisfaction_level.mean().plot(kind='bar')

data.groupby('left').average_montly_hours.mean().plot(kind='bar')
data.groupby('left').number_project.mean().plot(kind='bar')
sns.factorplot('number_project','average_montly_hours',hue='left',data=data )
data.groupby('left').time_spend_company.mean().plot(kind='bar')
sns.factorplot('time_spend_company','average_montly_hours',hue='left',data=data,size=5)
fig, axs = plt.subplots(ncols=2,figsize=(10,6))

data.groupby('left').Work_accident.mean().plot(kind='bar',ax = axs[0])

data.groupby('left').promotion_last_5years.mean().plot(kind='bar',ax=axs[1])

sns.factorplot('promotion_last_5years','time_spend_company',hue = 'left',data=data,size=6)
data.salary.value_counts().plot(kind='bar')
data.groupby('salary').left.mean().plot(kind='bar')
data.groupby('left').last_evaluation.mean().plot(kind='bar')
data.groupby('sales').left.mean().plot(kind='bar')
data.sales.value_counts()
old_employee = data['sales'][data['time_spend_company']>=5]

old_employee.value_counts()

old_employee_left = old_employee[data['left']==1]

old_employee_left.value_counts()
ratio_old_sales_left = float(old_employee_left.value_counts()[0])/float(data.sales.value_counts()[0])

ratio_old_technical_left = float(old_employee_left.value_counts()[1])/float(data.sales.value_counts()[1])

ratio_old_support_left  = float(old_employee_left.value_counts()[2])/float(data.sales.value_counts()[2])

ratio_old_IT_left = float(old_employee_left.value_counts()[3])/float(data.sales.value_counts()[3])

ratio_old_prodmgn_left = float(old_employee_left.value_counts()[4])/float(data.sales.value_counts()[4])

ratio_old_marketing_left = float(old_employee_left.value_counts()[5])/float(data.sales.value_counts()[5])

ratio_old_hr_left = float(old_employee_left.value_counts()[6])/float(data.sales.value_counts()[6])

raio_old_accounting_left = float(old_employee_left.value_counts()[7])/float(data.sales.value_counts()[7])

ratio_old_RandD_left = float(old_employee_left.value_counts()[8])/float(data.sales.value_counts()[8])

ratio_old_management_left = float(old_employee_left.value_counts()[9])/float(data.sales.value_counts()[9])



print ('Ratio of Old Employees leaving from Sales: ',ratio_old_sales_left)

print ('Ratio of Old Employees leaving from Technical: ',ratio_old_technical_left)

print ('Ratio of Old Employees leaving from Support: ',ratio_old_support_left)

print ('Ratio of Old Employees leaving from IT: ',ratio_old_IT_left)

print ('Ratio of Old Employees leaving from Product Management: ',ratio_old_prodmgn_left)

print ('Ratio of Old Employees leaving from Marketing: ',ratio_old_marketing_left)

print ('Ratio of Old Employees leaving from HR: ',ratio_old_hr_left)

print ('Ratio of Old Employees leaving from R&D: ',ratio_old_RandD_left)

print ('Ratio of Old Employees leaving from management: ',ratio_old_management_left)
