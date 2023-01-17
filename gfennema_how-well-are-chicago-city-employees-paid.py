#Get Plotly for pretty graphs
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()

#Regular Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
%matplotlib inline

#Read data and check contents
df = pd.read_csv('../input/chicago_employees.csv')
df.head()
#Convert currency into floats
df['Annual Salary'] = df[df.columns[6]].replace('[\$,]', '', regex=True).astype(float)
df['Hourly Rate'] = df[df.columns[7]].replace('[\$,]', '', regex=True).astype(float)

#Issolate full time salary employees
salary = df[df['Salary or Hourly']=='Salary']
salary = salary[salary['Full or Part-Time']=='F']

#Check the data types
salary.dtypes
#Median salary for a full time employee
print('Median Full Time Salary $'+str(salary['Annual Salary'].median()))
#Get median salary by department
medians = salary[['Department','Annual Salary']]
medians = medians.groupby(['Department']).median().sort_values('Annual Salary')

#Plot median salary using cufflinks
medians.iplot(kind='bar',margin=(50,50,180,80), title='Median Salary by Department')
#Get total amount spent on salaries by department
total = salary[['Department','Annual Salary']]
total = total.groupby(['Department']).sum().sort_values('Annual Salary')
total['Dep'] = total.index

#Plot using cufflinks
total.iplot(kind='pie', values='Annual Salary'
            ,labels='Dep',textposition='none',title = 'Sum of Salaries by Department')
#Top 10 salaries
salary.sort_values('Annual Salary',ascending=False).head(10)
#Isolate hourly employees
hourly = df[df['Salary or Hourly']=='Hourly']
#Typical hours per week multiplied by 52 to get estimated hours per year
hourly['Annual Hours'] = hourly['Typical Hours']*52
#Estimated hours per year times hourly rate give annual estimated total
hourly['Annual Total'] = hourly['Annual Hours']*hourly['Hourly Rate']
hourly.head()
print('Median Hourly Annual Total - $'+str(hourly['Annual Total'].median()))
print('Median Salary Annual Total - $'+str(salary['Annual Salary'].median()))
Hourly_vs_Salary = df.groupby(['Salary or Hourly']).count()
Hourly_vs_Salary['label'] = Hourly_vs_Salary.index
Hourly_vs_Salary.iplot(kind='pie', labels='label',values='Name'
                       ,title='Number of Salary vs. Hourly Employees')
#Number of hourly employees by department
hourly_count = hourly.groupby(['Department']).count()

#Plot with cufflinks
hourly_count['Name'].iplot(kind='bar',margin=(80,80,150,150), title='Number of Hourly Worker by Department')
#Median hourly rate by department
hourly_rate = hourly.groupby(['Department']).mean()

#Plot with cufflinks
hourly_rate['Hourly Rate'].iplot(kind='bar',title = 'Average Hourly Rate by Department',margin=(80,80,150,150))