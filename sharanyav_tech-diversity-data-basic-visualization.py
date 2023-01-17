#Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
#Read CSV, replace null values, and change data types
data = pd.read_csv('../input/Reveal_EEO1_for_2016.csv')
data['gender'] = data.gender.replace(np.NaN, 'All')
data = data.replace('na',0)
data['count'] = data['count'].astype(int)
#Subset data

#Create a table that just shows job types broken down by gender and race (removing totals)
bygender = data.loc[data['gender'] != 'All']
bygender = bygender.loc[data['job_category'] != 'Totals']
bygender = bygender.loc[data['job_category'] != 'Previous_totals']

#Create another table that displays totals for all job types
overalltotal = data.loc[data['gender'] == 'All']
del overalltotal['gender']
del overalltotal['race']
#create pivot table to show number of employees by gender and company
pivot = pd.pivot_table(bygender,index = 'company', columns = 'gender', values='count',aggfunc=sum)
pivot['percfemale'] = pivot['female']/(pivot['female']+pivot['male'])
pivot = pivot.sort_values('percfemale', ascending=False)
print(pivot)
#Create X axis and Y axes
x = list(pivot.index)
fem = pivot['female']
male = pivot['male']
percfemale = fem/(fem+male)
#Create bar graph to show which companies have the highest percentage of female employees
ind = np.arange(len(x))
width = 0.35
fig, ax = pl.subplots(figsize=(20,10))
rects1 = ax.bar(ind - width/2, percfemale, width, 
                color='blue', label='Percent of Female Employees')
ax.set_ylabel('Percent of employees')
ax.set_title('Employees by Gender')
ax.set_xticks(ind)
ax.set_xticklabels(x)
ax.legend()
pl.show()
#create scatter chart with line to show how these companies are doing, compared to other companies their size
logfem = np.log(fem)
logmale = np.log(male)
pl.figure(figsize=(20,10))
pl.scatter(logfem, logmale, alpha=0.5)
for i in range (0,len(x)):
    xy=(logfem[i],logmale[i])
    pl.annotate(x[i],xy)
pl.plot([0, 11], [0, 11], 'k-', color = 'r')
pl.xlim(4,11)
pl.ylim(4,11)
pl.ylabel("Male Employees")
pl.xlabel("Female Employees")
pl.title("Distribution of Male and Female Employees")
pl.show()