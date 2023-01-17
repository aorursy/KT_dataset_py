import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

import operator

print(os.listdir("../input"))
df = pd.read_csv('../input/breakdownlist.csv', delimiter=';', index_col='id')
df1= df.dropna(axis=0)
df.head()
nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
nRow, nCol = df1.shape

print(f'After dropping rows with missing data, there are {nRow} rows and {nCol} columns')
cleaning_sum = df1.loc[df1['cause'] == 'Cleaning' , 'total'].sum()

breakdown_sum = df1.loc[df1['cause'] == 'breakdown' , 'total'].sum()

trouble_sum = df1.loc[df1['cause'] == 'trouble' , 'total'].sum()

other_sum = df1.loc[df1['cause'] == 'other' , 'total'].sum()
x1 = ['Cleaning','breakdown','trouble','other']

y1 = [cleaning_sum, breakdown_sum, trouble_sum, other_sum]
plt.bar(x1,y1, color='green')

plt.ylabel('down time (min)')

plt.xlabel('cause')

plt.title('down time vs. cause')
machine_log_breakdown={}

for index, row in df1.iterrows():  

    if row['cause'] == 'breakdown':

        if row['cncmachine'] in machine_log_breakdown:

            machine_log_breakdown[row['cncmachine']]=machine_log_breakdown[row['cncmachine']]+row['total']

        else:

            machine_log_breakdown[row['cncmachine']]=row['total']
sorted_log_breakdown = sorted(machine_log_breakdown.items(), key=operator.itemgetter(1),reverse=True)
x2=[]

y2=[]

for i in range(0,10,1):

    x2.append(sorted_log_breakdown[i][0])

    y2.append(sorted_log_breakdown[i][1])
plt.bar(x2,y2, color='r')

plt.ylabel('breakdown time (min)')

plt.xlabel('machine name')

plt.title('down time vs. machine')
personnel_log={}

for index, row in df1.iterrows():  

    if row['personnelid'] in personnel_log:

        if row['cause'] == 'Cleaning':

            personnel_log[row['personnelid']][0]=personnel_log[row['personnelid']][0]+row['total']

        elif row['cause'] == 'breakdown':

            personnel_log[row['personnelid']][1]=personnel_log[row['personnelid']][1]+row['total']

        else:

            personnel_log[row['personnelid']][2]=personnel_log[row['personnelid']][2]+row['total']

    else:

        if row['cause'] == 'Cleaning':

            personnel_log[row['personnelid']]=[row['total'],0,0]

        elif row['cause'] == 'breakdown':

            personnel_log[row['personnelid']]=[0,row['total'],0]

        else:

            personnel_log[row['personnelid']]=[0,0,row['total']]
hour_sum={}

for key,value in personnel_log.items():

    print(key,':', value)

    hour_sum[key] = np.sum(value)
sorted_sum_hours = sorted(hour_sum.items(), key=operator.itemgetter(1),reverse=True)
for i in range(0,10,1):

    print(sorted_sum_hours[i][0],':', sorted_sum_hours[i][1])
x3=[]



y3_1=[]

y3_2=[]

y3_3=[]



for i in range(0,10,1):

    employee_id = sorted_sum_hours[i][0]

    

    # retrieving total Cleaning time for employee i

    y3_1.append(personnel_log[employee_id][0]) 

    # retrieving total breakdown time for employee i

    y3_2.append(personnel_log[employee_id][1])

    # retrieving total other time for employee i

    y3_3.append(personnel_log[employee_id][2])

    

    x3.append(str(sorted_sum_hours[i][0]))
plt.bar(x3,y3_1, color='b', label='Cleaning')

plt.bar(x3,y3_2, color='r', label='breakdown')

plt.bar(x3,y3_3, color='g',label='other')

plt.ylabel('breakdown time (min)')

plt.xlabel('personnel id')

plt.title('down time vs. employee id')

plt.legend()

plt.show()
