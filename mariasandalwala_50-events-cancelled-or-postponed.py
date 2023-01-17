import pandas as pd

data = pd.read_csv("../input/50-large-scale-events-cancelled-covid19/Untitled spreadsheet - Sheet1.csv")

from matplotlib import pyplot as plt
group1 = data.loc[data['Attendance'] >= 300000]

group2 = data.loc[(data['Attendance'] >= 200000) & (data['Attendance'] < 300000)]

group3 = data.loc[(data['Attendance'] >= 100000) & (data['Attendance'] < 200000)]

group4 = data.loc[(data['Attendance'] < 100000)]
import numpy as np

arr=[0,1,2,3]

arr[3] = 100*(group1.loc[group1['Status'] == 'Cancelled'].Status.count()/group1.Status.count())

arr[2] = 100*(group2.loc[group2['Status'] == 'Cancelled'].Status.count()/group2.Status.count())

arr[1] = 100*(group3.loc[group3['Status'] == 'Cancelled'].Status.count()/group3.Status.count())

arr[0] = 100*(group4.loc[group4['Status'] == 'Cancelled'].Status.count()/group4.Status.count())



arrp = [100-arr[0],100-arr[1],100-arr[2],100-arr[3]]

ind = np.arange(4)

p1 = plt.bar(ind,arr,width=0.5)

p2 = plt.bar(ind,arrp,width=0.5,bottom=arr)



plt.ylabel('Percentage')

plt.xlabel('Attendance')

plt.title('Scores by group and gender')

plt.xticks(ind, ('<1 Lakh', '1-2 Lakh', '2-3 Lakh', '>3 Lakh'))

plt.legend((p1[0], p2[0]), ('Cancelled', 'Postponed'))

plt.show()