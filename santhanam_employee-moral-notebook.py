# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting and graphing

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





hr_data = pd.read_csv('../input/HR_comma_sep.csv')
hr_data.describe()

hr_data.head()
past_employee = hr_data[hr_data['left'] > 0]

current_employee = hr_data[hr_data['left'] <= 0]



print (past_employee.shape)

print (current_employee.shape)
print (past_employee['sales'].unique())

# sales,technical,support

# maj_att_department = past_employee[(past_employee['sales']=='sales')or(past_employee['sales']=='technical')]

# maj_att_department.shape

m_att_dep = past_employee[past_employee['sales']!='accounting'][past_employee['sales']!='hr'][past_employee['sales']!='management']

sns.countplot(x='sales',data=m_att_dep)
print(m_att_dep.shape)

print(m_att_dep.columns)
sns.factorplot(x="salary", y="satisfaction_level", hue="sales", data=m_att_dep);
sns.pointplot(y="last_evaluation", x="number_project",  data=m_att_dep)
past_employee['number_project'].describe()
sns.boxplot(x="salary", y="number_project", data=past_employee);
sns.countplot(x="sales", data=current_employee);
sns.pointplot(x="sales", y="number_project", hue="salary",  data=current_employee)
past_employee.columns
current_employee['sales'].unique()


sns.pointplot(y="time_spend_company", x="salary", data=current_employee[current_employee['sales']=='marketing'])
sns.pointplot(y="last_evaluation", x="salary", hue="sales", data=past_employee[past_employee['sales']=='IT'])
sns.pointplot(y="satisfaction_level", x="salary",  data=current_employee)
sns.pointplot(y="satisfaction_level", x="salary",  data=past_employee)
hr_data.columns