import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import json

from pandas import DataFrame

import ijson

from pandas.io.json import json_normalize

with open('../input/200620190952.json') as f:

    df = json.load(f)
data=pd.DataFrame([i for i in df['8rAw8mdXq8TYHSrPaNflfmkyP4i1']['Smartphone']['AppTimeUsage']])
print('list of program used by a user ')

print(data)
appname=[]

time=[]

for k,v in df['8rAw8mdXq8TYHSrPaNflfmkyP4i1']['Smartphone']['AppTimeUsage'].items():

    appname.append(k)

    count=0

    for kk,vv in v.items():

        for kkk,vvv in vv.items():

            if(kkk=="Time Used"):

                count+=vvv

    time.append(count)

time=np.array(time)

time=(time//60000)+1

appname=np.array(appname)

ind=np.arange(0,26)+1

user1=pd.DataFrame({'appname':appname,'time':time},index=ind)  

    
appname2=[]

time2=[]

for k,v in df['BVIf6qW2amPIBuvS2jhlNVNF1xy2']['Smartphone']['AppTimeUsage'].items():

    appname2.append(k)

    count=0

    for kk,vv in v.items():

        for kkk,vvv in vv.items():

            if(kkk=="Time Used"):

                count+=vvv

    time2.append(count)

time2=np.array(time2)

#print(time2)

time2=(time2//60000)+1

appname2=np.array(appname2)

#ind=np.arange(0,26)+1

#user2=pd.DataFrame({'appname':appname2,'time':time2},index=ind)  
print(user1)

print('time in minutes')
l=[]

#print(user1.iloc[0])

for i in range(user1.shape[0]):

    for j in range(user1.iloc[i][1]):

        l.append(user1.iloc[i][0])

    #print('hi')

l=np.array(l)

ind=np.arange(0,len(l))+1

user1=pd.DataFrame({'id':ind,'data':l},index=ind)  

    



sns.set(style="darkgrid")

plt.figure(figsize=(10, 5))

plt.subplot(1,1,1)

plt.plot(user1.data.values,color='blue',label='time spent')

plt.show()

print('User1 Statistics Please double tap to zoom in to look the app name')

plt.figure(figsize=(40, 25))

ax=plt.subplot(1,1,1)



left = np.arange(0,len(time))+1

  

# heights of bars 

height = time

  

# labels for bars 

tick_label = appname

  

# plotting a bar chart 

plt.bar(left, height, tick_label = tick_label, 

        width = 0.8, color = ['red', 'green','blue','yellow']) 

  

# naming the x-axis 

plt.xlabel('App Usage') 

# naming the y-axis 

plt.ylabel('Time Used') 

# plot title 

plt.title('User1 Usage Statistics') 

  

# function to show the plot 

plt.show() 
print('User2 Statistics Please double tap to zoom in to look the app name')

plt.figure(figsize=(60, 25))

ax=plt.subplot(1,1,1)



left = np.arange(0,len(time2))+1

  

# heights of bars 

height = time2

  

# labels for bars 

tick_label = appname2

  

# plotting a bar chart 

plt.bar(left, height, tick_label = tick_label, 

        width = 0.8, color = ['red', 'green','blue','yellow','orange','pink']) 

  

# naming the x-axis 

plt.xlabel('App Usage') 

# naming the y-axis 

plt.ylabel('Time Used') 

# plot title 

plt.title('User2 Usage Statistics') 

  

# function to show the plot 

plt.show() 
appname3=[]

time3=[]

for k,v in df['lmF7VgWqllgKEBUuQOa6lrQPlQw2']['Smartphone']['AppTimeUsage'].items():

    appname3.append(k)

    count=0

    for kk,vv in v.items():

        for kkk,vvv in vv.items():

            if(kkk=="Time Used"):

                count+=vvv

    time3.append(count)

time3=np.array(time3)

time3=(time3//60000)+1

appname3=np.array(appname3)

print('User3 Statistics Please double tap to zoom in to look the app name')

plt.figure(figsize=(60, 25))

ax=plt.subplot(1,1,1)



left = np.arange(0,len(time3))+1

  

# heights of bars 

height = time3

  

# labels for bars 

tick_label = appname3

  

# plotting a bar chart 

plt.bar(left, height, tick_label = tick_label, 

        width = 0.8, color = ['red', 'green','blue','yellow','orange','pink']) 

  

# naming the x-axis 

plt.xlabel('App Usage') 

# naming the y-axis 

plt.ylabel('Time Used') 

# plot title 

plt.title('User3 Usage Statistics') 

  

# function to show the plot 

plt.show() 
appname4=[]

time4=[]

for k,v in df['r3jYUgZHrBTTlsGDy2VPrKYIL9j2']['Smartphone']['AppTimeUsage'].items():

    appname4.append(k)

    count=0

    for kk,vv in v.items():

        for kkk,vvv in vv.items():

            if(kkk=="Time Used"):

                count+=vvv

    time4.append(count)

time4=np.array(time4)

time4=(time4//60000)+1

appname4=np.array(appname4)



print('User4 Statistics Please double tap to zoom in to look the app name')

plt.figure(figsize=(60, 25))

ax=plt.subplot(1,1,1)



left = np.arange(0,len(time4))+1

  

# heights of bars 

height = time4

  

# labels for bars 

tick_label = appname4

  

# plotting a bar chart 

plt.bar(left, height, tick_label = tick_label, 

        width = 0.8, color = ['red', 'green','blue','yellow','orange','pink']) 

  

# naming the x-axis 

plt.xlabel('App Usage') 

# naming the y-axis 

plt.ylabel('Time Used') 

# plot title 

plt.title('User4 Usage Statistics') 

  

# function to show the plot 

plt.show() 