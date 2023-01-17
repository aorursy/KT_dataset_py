# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
data.head()
df=data[['Country','State','City','Start_Time','End_Time','Severity']]
df.head()
df.Severity.value_counts()
import matplotlib.pyplot as plt
df.State.value_counts()[:10].plot(kind='pie')
plt.axis('equal')
plt.show()
print("Total number of States in the dataset = "+str(len(df.State.unique())))
print('From the above plotting it is clear that California(CA) has most number of accidents.')
temp = df['Start_Time'].str.split(" ",expand=True)
df['date'] = temp[0]
df['Time'] = temp[1]
df.head()
from tqdm import tqdm
num = []
for i in range(0,24):
    num.append(i)
i=0
j=1
count = []

for i in tqdm(range(len(num)-1)):
    #print("i "+str(num[i]))
    
    #print("j "+str(num[j]))
    
    if i < 9:
       
        #print('0'+str(i)+":"+'00'+':'+"00")
        hour_start = '0'+str(i)+":"+'00'+':'+"00"
        #print('0'+str(j)+":"+'00'+':'+"00")
        hour_end = '0'+str(j)+":"+'00'+':'+"00"
        #print(" ")
        #print(hour_start)
        #print(hour_end)
        #print(len(df[(df['Time']>=hour_start)&(df['Time']<=hour_end)]))
        count.append(len(df[(df['Time']>=hour_start)&(df['Time']<=hour_end)]))
    
    elif i==9:
            hour_start = '0'+str(i)+":"+'00'+':'+"00"
            hour_end = str(j)+":"+'00'+':'+"00"
            #print(hour_start)
            #print(hour_end)
            #print(len(df[(df['Time']>=hour_start)&(df['Time']<=hour_end)]))
            count.append(len(df[(df['Time']>=hour_start)&(df['Time']<=hour_end)]))
            
            
    else:
         #print(str(i)+":"+'00'+':'+"00")
        hour_start = str(i)+":"+'00'+':'+"00"
        #print(str(j)+":"+'00'+':'+"00")
        hour_end = str(j)+":"+'00'+':'+"00"
        #print(" ")
        #print(hour_start)
        #print(hour_end)
        #print(len(df[(df['Time']>=hour_start)&(df['Time']<=hour_end)]))
        count.append(len(df[(df['Time']>=hour_start)&(df['Time']<=hour_end)]))
        
    i = j;
    j = i+1;
    #print("")
    
hour = []
for i in range(len(count)):
    hour.append(i)
X = hour
y = count

import seaborn as sns
sns.set(style="whitegrid")
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax = sns.barplot(x=X, y=count)
print('we can cleary see that in the morning hours that is from 8 AM - 9 AM there are more accidents happening')

