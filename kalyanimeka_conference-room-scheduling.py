# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

## Excel sheet that stores room_no, floor_no, max_no_of_people in the room ##



df=pd.read_excel('../input/CONFERENCEROOMS.xlsx')

df

newind = 'df1 df2 df3 df4 df5 df6'.split()

df['CASE'] = newind

df.set_index('CASE')

##Sort the rows with given number of people##

newdf = df[(df['MAX_NO']>5)]

newdf

newdf.set_index('CASE')

## Sort the rooms with nearest floor##

n=8

newdf2 = newdf[(newdf['FLOOR']==(n))] 

newdf3 = newdf[(newdf['FLOOR']==(n+1))] 

newdf4 = newdf[(newdf['FLOOR']==(n-1))] 

newdf5 = newdf2.append(newdf3)

newdf6 = newdf5.append(newdf4)

newdf7 = newdf6.sort_index(ascending=True)

newdf7

##Choosing the start and end date to check the timeslots

start = pd.to_timedelta('10:30:00')

end = pd.to_timedelta('11:30:00')

## Database for different timesots from given input for each room##



df1 = pd.DataFrame(

{'Slot_no':[1,2],

'start_time':['9:00:00','14:30:00'],

'end_time':['9:15:00','15:00:00']})

f1 = df1.reindex_axis(['Slot_no','start_time','end_time'], axis=1)                 

df1['start_time'] = pd.to_timedelta(df1['start_time'])

df1['end_time'] = pd.to_timedelta(df1['end_time'].replace('0:00:00', '24:00:00'))

mask1 = df1['start_time'].between(start, end) | df1['end_time'].between(start,end)

mask1



df2 = pd.DataFrame(

{'Slot_no':[1,2],

'start_time':['10:00:00','14:30:00'],

'end_time':['11:00:00','15:00:00']})

f2 = df2.reindex_axis(['Slot_no','start_time','end_time'], axis=1)                 

df2['start_time'] = pd.to_timedelta(df2['start_time'])

df2['end_time'] = pd.to_timedelta(df2['end_time'].replace('0:00:00', '24:00:00'))

mask2 = df2['start_time'].between(start, end) | df2['end_time'].between(start,end)

mask2

    

df3 = pd.DataFrame(

{'Slot_no':[1,2],

'start_time':['11:30:00','17:00:00'],

'end_time':['12:30:00','17:30:00']})

f3 = df3.reindex_axis(['Slot_no','start_time','end_time'], axis=1)                 

df3['start_time'] = pd.to_timedelta(df3['start_time'])

df3['end_time'] = pd.to_timedelta(df3['end_time'].replace('0:00:00', '24:00:00'))

mask3 = df3['start_time'].between(start, end) | df3['end_time'].between(start,end)

mask3



df4 = pd.DataFrame(

{'Slot_no':[1,2,3],

'start_time':['9:30:00','12:00:00','15:15:00'],

'end_time':['10:30:00','12:15:00','16:15:00']})

f4 = df4.reindex_axis(['Slot_no','start_time','end_time'], axis=1)                 

df4['start_time'] = pd.to_timedelta(df4['start_time'])

df4['end_time'] = pd.to_timedelta(df4['end_time'].replace('0:00:00', '24:00:00'))

mask4 = df4['start_time'].between(start, end) & df4['end_time'].between(start,end)

mask4



df5 = pd.DataFrame(

{'Slot_no':[1,2],

'start_time':['9:00:00','11:00:00'],

'end_time':['14:00:00','16:00:00']})

f5 = df5.reindex_axis(['Slot_no','start_time','end_time'], axis=1)                 

df5['start_time'] = pd.to_timedelta(df5['start_time'])

df5['end_time'] = pd.to_timedelta(df5['end_time'].replace('0:00:00', '24:00:00'))

mask5 = df5['start_time'].between(start, end) | df5['end_time'].between(start,end)

mask5



df6 = pd.DataFrame(

{'Slot_no':[1,2,3],

'start_time':['10:30:00','13:30:00','16:30:00'],

'end_time':['11:30:00','15:30:00','17:30:00']})

f6 = df6.reindex_axis(['Slot_no','start_time','end_time'], axis=1)                 

df6['start_time'] = pd.to_timedelta(df6['start_time'])

df6['end_time'] = pd.to_timedelta(df6['end_time'].replace('0:00:00', '24:00:00'))

mask6 = df6['start_time'].between(start, end) | df6['end_time'].between(start,end)

mask6

##To observe the time slot that matches for the required conference room scheduling##

for row in newdf7['CASE']:

    if row == 'df1':

        print (mask1)

    elif row == 'df2':

        print (mask2)

    elif row == 'df3':

        print (mask3)

    elif row == 'df4':

        print (mask4)

    elif row == 'df5':

        print (mask5)

    elif row == 'df6':

        print (mask6)

    else:

        print ('Not Available')

##EXTRA CREDIT:

## AS the meeting time is only for one hour, splitting the conference room for less than 30 minutes would'nt be a suggested idea

start1 = pd.to_timedelta('10:30:00')

end1 = pd.to_timedelta('11:00:00')

start2 = pd.to_timedelta('11:00:00')

end2 = pd.to_timedelta('11:30:00')



#By splitting the time slot and running the above code again, we can obtain the available timeslots and conference rooms to schedule the meeting in two different rooms. Certainly we can also split them into n slots based on our requirement. 
