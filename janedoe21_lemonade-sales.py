import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/Lemonade.csv')
average = np.average(dataset['Sales'])
print('average sales = ', average)
lower = (dataset.loc[dataset['Sales'] < average])
print(lower)
x = dataset['Sales']
y = dataset['Temperature']
plt.scatter(x,y)
plt.ylabel('Temperature')
plt.xlabel('Sales')
sunday = (dataset[dataset['Day'] == 'Sunday'])
monday = (dataset[dataset['Day'] == 'Monday'])
tuesday = (dataset[dataset['Day'] == 'Tuesday'])
wednesday = (dataset[dataset['Day'] == 'Wednesday'])
thursday = (dataset[dataset['Day'] == 'Thursday'])
friday = (dataset[dataset['Day'] == 'Friday'])
saturday = (dataset[dataset['Day'] == 'Saturday'])

avgsun = np.average(sunday['Sales'])
avgmon = np.average(monday['Sales'])
avgtue = np.average(tuesday['Sales'])
avgwed = np.average(wednesday['Sales'])
avgthu = np.average(thursday['Sales'])
avgfri = np.average(friday['Sales'])
avgsat = np.average(saturday['Sales'])

print('average sales on Sunday = ', avgsun)
print('average sales on Monday = ', avgmon)
print('average sales on Tuesday = ', avgtue)
print('average sales on Wednesday = ', avgwed)
print('average sales on Thursday = ', avgthu)
print('average sales on Friday = ', avgfri)
print('average sales on Saturday = ', avgsat)

days = ('Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat')
y_pos = np.arange(len(days))
avg = [avgsun, avgmon, avgtue, avgwed, avgthu, avgfri, avgsat]
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, days)
plt.ylabel('Average Sales')
plt.title('Average sales on each day')
plt.show()