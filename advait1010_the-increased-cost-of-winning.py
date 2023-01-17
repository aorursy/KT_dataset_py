
from __future__ import division
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import csv


#always use header=0 when you know row 0 is the header row

#Ignore warnings
warnings.filterwarnings('ignore')

#First Find Winners' budgets

#Get winning teams since 1985

#Read through team.csv
data = pd.read_csv('../input/team.csv',header = 0)
#Data is of type pandas.core.frame.DataFrame


d1 = data[['year','team_id','ws_win']]

#We want only WS winners
d2 = d1[d1['ws_win']=='Y']

#Inintialize Payroll 
Payroll = []


#Get Salaries
salary = pd.read_csv('../input/salary.csv',header = 0)
s1 = salary[['year','team_id','salary']]


#Function to find total pay each year
def FindPay(years):
	d3 = d2[d2['year'] == years]
	s2 = s1[s1['year'] == years]
	m = pd.merge(d3,s2, on = ['year','team_id'], how = 'inner')
	Payroll.append(int(sum(m['salary'])/1000000))


#For all years
for i in range(1985,2016):
	FindPay(i)


# Now we find the average

avg = pd.read_csv('../input/salary.csv',header = 0)
a1 = avg[['year','salary']]
average = []


def FindAverage(years):
	a2 = a1[a1['year'] == years]
	average.append(int(sum(a2['salary'])/30000000))



for i in range(1985,2016):
	FindAverage(i)

yrs = []
for i in range(1985,2016):
	yrs.append(i)	



f = plt.figure(1)
plt.plot(yrs, Payroll, label = 'Winners', marker = 'o', color = 'r',linestyle = '-.')
plt.plot(yrs, average, label = 'Average', marker = 's', color = 'b', linestyle = '--')
plt.xlabel('Years')
plt.ylabel('Total Payroll')
plt.title('Average and Winner Budgets')
plt.legend()
plt.show()

#Finding average of salary differences, by average

diff = []
perc = []

def Diff(years):
	d3 = d2[d2['year'] == years]
	s2 = s1[s1['year'] == years]
	m = pd.merge(d3,s2, on = ['year','team_id'], how = 'inner')
	m = sum(m['salary'])/1000000

	a2 = a1[a1['year'] == years]
	n = sum(a2['salary'])/30000000

	return m-n,n



for i in range(1985,2016):
	x,y = Diff(i)
	pct = np.divide(x,y)
	pct *= 100
	if i != 1994:
		diff.append(int(pct))
	else:
		diff.append(0)
		


# Adjusted for 1994, no WS



g = plt.figure(2)
plt.plot(yrs, diff,marker = 'o', color = 'g')
plt.xlabel('Years')
plt.ylabel('Percentage differnece in Payrolls')
plt.title('Percentage Advantage of Winners') 
plt.legend()
plt.show()

    



