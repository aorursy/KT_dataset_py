import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

%matplotlib inline

import plotly

plotly.offline.init_notebook_mode()

import plotly.graph_objects as go
pd.set_option('display.max.rows',None)  # for displaying whole dataset

df = pd.read_csv('../input/2010-report-to-congress-on-white-house-staff/2010_Report_to_Congress_on_White_House_Staff (1).csv')

     # reading the CSV file

df
df.head()   # top 5 rows from the dataset
df.tail()   # bottom 5 rows from the dataset
df.shape   # indicates the dimensional view of Dataset in terms of (rows,columns)
df.columns    # indicates the column labels
df.info()   # info about the dataframe including index dtypes and columns, non-null values and memory usage.

df.describe()   # Calculation of Statistical Data
df.describe(include=['object', 'bool'])   # Calculation of Statistical Data wrt Objects and Boolean
df[df.Salary==df.Salary.max()] #Details of people with maximum salary
df[df.Salary==df.Salary.min()] #Details of people with minimum salary
df[df['Salary']>0].min()  # Minimum Salary greater than zero
df.Salary.mean()    # Mean Salary


df1 = px.data.tips()

fig = px.histogram(df, x=df.Salary,title='Count of employees in each salary range')



fig.show()
#  function returns the Series containing counts of unique values.

Number_of_Employee=df['Employee Status'].value_counts()  

print('Number of Employee, Detailee, Employee (part-time)\n',Number_of_Employee)
plt.title('No. of people in each employee status')

plt.xlabel('Count')

Number_of_Employee.plot(kind='barh',color='#4EE2EC',figsize = (20, 8))  # Bar Plot

plt.show()
labels = 'Employee ', 'Detailee', 'Employee (part-time)'     # Doughnut Plot

sizes = [437, 31, 1]

colors = ['#DF9D9D','#802A2A','#330000']

explode = (0.7, 0.7, 0.7)

plt.title('Percentage of people in each employee status')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.2f%%', shadow=True,radius=5,pctdistance=0.8)

centre_circle = plt.Circle((0,0),radius=2,color='white', fc='white')

fig = plt.gcf()

fig.set_size_inches(15,8) 

fig.gca().add_artist(centre_circle)

plt.axis('equal')

plt.show() 
plt.title('Mean salary for each employee status')

sns.barplot(x=df['Employee Status'],y=df.Salary,estimator=np.mean)   # Seaborn Bar Plot
df5=df.loc[:,['Employee Status','Salary',]]#Selection of column range

X=df5[df5['Employee Status']=='Detailee']#Detailee

Y=df5[df5['Employee Status']=='Employee']#Employee

Z=df5[df5['Employee Status']=='Employee (part-time)']#Employee (part-time)

x=X['Salary'].sum()

y=Y['Salary'].sum()

z=Z['Salary'].sum()

print('Total salary to Detailee section: ',x)

print('Total salary to Employee section: ',y)

print('Total salary to Employee (part-time) section: ',z)
Employee_Status = ['Detailee', 'Employee', 'Employee (part-time)']   # Pie Chart

Salary_status = [x,y,z]

#explode = (0.2, 0.3, 0.3)

fig = plt.figure(figsize =(10, 3))

plt.pie(Salary_status, labels = Employee_Status, colors=['#EFEFE8FF','#95DBE5FF','#000000FF'],autopct='%0.3f%%',shadow=True,radius=3,pctdistance=0.7,explode=explode)

plt.show()
# Swarm Plot

sns.set(style="darkgrid")                

fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Number of Employees for each salary range with respect to their Employee Status')

sns.swarmplot(x='Employee Status', y='Salary', data=df,size=15,ax=ax)
df8= df[df['Employee Status'] =='Detailee']

df8
# Seaborn Cat Plot

sns.set(style="darkgrid")

sns.catplot('Employee Status','Salary',data=df8,hue='Employee Name',kind='bar',height=8)

plt.xlabel('Employee Status - Detailee')

plt.title('Salary Range for Detailee')
Detailee=df[df['Employee Status']=='Detailee'].sort_values(by=['Salary'],ascending=False).head(1)

print('Maximum salary in Detailee section is: ')

Detailee
Employee=df[df['Employee Status']=='Employee'].sort_values(by=['Salary'],ascending=False).head(n=1)

print('Maximum salary in Employee section is: ')

Employee
Employee_part_time=df[df['Employee Status']=='Employee (part-time)'].sort_values(by=['Salary'],ascending=False).head(n=1)

print('Maximum salary in Employee (part-time) section is: ')

Employee_part_time
Detailee=df[df['Employee Status']=='Detailee'].sort_values(by=['Salary'],ascending=True).head(n=1)

print('Minimum salary in Detailee section is: ')

Detailee
Employee=df[df['Employee Status']=='Employee'].sort_values(by=['Salary'],ascending=True)

df7=Employee[Employee['Salary']>0].head(n=1)

print('Minimum salary in Employee section is: ')

df7
A=df['Position Title'].value_counts()      # Series Containing counts of Unique Values

Number_of_Position_title=A.sort_values(ascending=False)

print('Number of Position Titles\n',Number_of_Position_title)
fig=px.bar(data_frame=A,y=Number_of_Position_title, height=400,width=2300,color=Number_of_Position_title,labels={'y':'Count'},title='Count of employees for each position title' )

fig.show()    # Plotly Bar Graph
df2= df[df['Position Title']=='RECORDS MANAGEMENT ANALYST']

df2
sns.set(style="darkgrid")          # Seaborn Cat Plot 

sns.catplot('Employee Status','Salary',data=df2,hue='Employee Name',kind='bar',height=8)

plt.xlabel('Position Title-Record management Analyst')

plt.title('Range of salaries for Record Management Analyst')