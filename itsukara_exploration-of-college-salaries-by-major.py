import numpy as np

import pandas as pd

import matplotlib

from matplotlib import pyplot as plt
df1 = pd.read_csv('../input/degrees-that-pay-back.csv')    #by major (50)              -- starting, median, percentile salaries

df2 = pd.read_csv('../input/salaries-by-college-type.csv') #by uni (269) / school type -- starting, median, percentile salaries

df3 = pd.read_csv('../input/salaries-by-region.csv')       #by uni (320) / region      -- starting, median, percentile salaries
df1.head()
df1.columns = ['major','bgn_p50','mid_p50','delta_bgn_mid','mid_p10','mid_p25','mid_p75','mid_p90']

df1.head()
type(df1['bgn_p50'][1])
dollar_cols = ['bgn_p50','mid_p50','mid_p10','mid_p25','mid_p75','mid_p90']



for x in dollar_cols:

    df1[x] = df1[x].str.replace("$","")

    df1[x] = df1[x].str.replace(",","")

    df1[x] = pd.to_numeric(df1[x])



df1.head()
df1['bgn_p50'].mean()
df1.describe()
df1.head()
df1.sort_values(by = 'bgn_p50', ascending = False, inplace=True)

df1.head()
df1 = df1.reset_index()

df1.head(10)
x = df1.index

y = df1['bgn_p50']

labels = df1.index



plt.scatter(x,y, color='g', label = 'Starting Median Salary')

plt.xticks(x, labels) 



plt.xlabel('Major')

plt.ylabel('US Dollars')

plt.title('Starting Median Salary by Major')

plt.legend()

plt.show()
x = df1.index

y = df1['bgn_p50']

labels = df1['major']

#labels = df1.index



plt.scatter(x,y, color='g', label = 'Starting Median Salary')

plt.xticks(x, labels, rotation = 'vertical') #rotation = 'vertical'



plt.xlabel('Major')

plt.ylabel('US Dollars')

plt.title('Starting Median Salary by Major')

plt.legend()

plt.show()
x = df1['bgn_p50'] #switch x and y labels

y = df1.index

labels = df1['major']

#labels = df1.index



plt.scatter(x, y, color='g', label = 'Starting Median Salary') 

plt.yticks(y, labels)



plt.xlabel('US $')

plt.ylabel('') #hide label

plt.title('Starting Median Salary by Major')

plt.legend()

plt.show()
x = df1['bgn_p50']

y = len(df1.index) - df1.index #swap high and low

labels = df1['major']



plt.scatter(x, y, color='g', label = 'Starting Median Salary')

plt.yticks(y, labels)



plt.xlabel('US $')

plt.ylabel('')

plt.title('Starting Median Salary by Major')

plt.legend()

plt.show()
fig = plt.figure(figsize=(8,12))



x = df1['bgn_p50']

y = len(df1.index) - df1.index

labels = df1['major']



plt.scatter(x, y, color='g', label = 'Starting Median Salary')

plt.yticks(y, labels)



plt.xlabel('US $')

plt.ylabel('')

plt.title('Starting Median Salary by Major')

plt.legend(loc=2) #move the legend

plt.show()
fig = plt.figure(figsize=(8,12))



x = df1['bgn_p50']

y = len(df1.index) - df1.index

labels = df1['major']



plt.scatter(x, y, color='#d6d6d6', label = 'Median Starting Salary')

plt.yticks(y, labels)



x3 = df1['mid_p50']

plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')



plt.xlabel('US $')

plt.ylabel('')

plt.title('Salary Information by Major')

plt.legend(loc=2) #move the legend

plt.show()
df2 = df1.sort_values(by = 'mid_p50', ascending = False)

df2 = df2.reset_index()



fig = plt.figure(figsize=(8,12))



x = df2['bgn_p50']

y = len(df2.index) - df2.index

labels = df2['major']



plt.scatter(x, y, color='#d6d6d6', label = 'Median Starting Salary')

plt.yticks(y, labels)



x3 = df2['mid_p50']

plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')



plt.xlabel('US $')

plt.ylabel('')

plt.title('Salary Information by Major')

plt.legend(loc=2) #move the legend

plt.show()
df2 = df1.sort_values(by = 'mid_p50', ascending = False)

df2 = df2.reset_index()



fig = plt.figure(figsize=(8,12))



x = df2['bgn_p50']

y = len(df2.index) - df2.index

labels = df2['major']



#plt.scatter(x, y, color='b', label = 'Median Starting Salary')

plt.yticks(y, labels)



x2 = df2['mid_p25']

plt.scatter(x2, y, color='#ecc833', label = '25th pct. Mid Career Salary')



x3 = df2['mid_p50']

plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')



x4 = df2['mid_p75']

plt.scatter(x4, y, color='#2b5bde', label = '75th pct. Mid Career Salary')



plt.xlabel('US $')

plt.ylabel('')

plt.title('Salary Information by Major')

plt.legend(loc=2) #move the legend

plt.show()
df2.index
df2 = df1.sort_values(by = 'mid_p50', ascending = False)

df2 = df2.reset_index()



fig = plt.figure(figsize=(8,12))



x = df2['bgn_p50']

y = len(df2.index) - df2.index + 1

labels = df2['major']



#plt.scatter(x, y, color='b', label = 'Median Starting Salary')

plt.yticks(y, labels)



x1 = df2['mid_p10']

plt.scatter(x1, y, color='#f7e9ad', label = '10th pct. Mid Career Salary')



x2 = df2['mid_p25']

plt.scatter(x2, y, color='#ecc833', label = '25th pct. Mid Career Salary')



x3 = df2['mid_p50']

plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')



x4 = df2['mid_p75']

plt.scatter(x4, y, color='#2b5bde', label = '75th pct. Mid Career Salary')



x5 = df2['mid_p90']

plt.scatter(x5, y, color='#a1b6f0', label = '90th pct. Mid Career Salary')



plt.xlabel('US $')

plt.ylabel('')

plt.title('Salary Information by Major')

plt.legend(loc='upper right', bbox_to_anchor=(1.42,.98)) #move the legend



plt.show()
df2 = df1.sort_values(by = 'mid_p50', ascending = False)

df2 = df2.reset_index()



fig = plt.figure(figsize=(8,12))

matplotlib.rc('grid', alpha = .5, color = '#e3dfdf')   #color the grid lines

matplotlib.rc('axes', edgecolor = '#67746A')           #color the graph edge

#matplotlib.rcParams.update(matplotlib.rcParamsDefault)  #this will reset default params if you need



x = df2['bgn_p50']

y = len(df2.index) - df2.index + 1

labels = df2['major']



#plt.scatter(x, y, color='b', label = 'Median Starting Salary')

plt.yticks(y, labels)



x1 = df2['mid_p10']

plt.scatter(x1, y, color='#f7e9ad', label = '10th pct. Mid Career Salary')



x2 = df2['mid_p25']

plt.scatter(x2, y, color='#ecc833', label = '25th pct. Mid Career Salary')



x3 = df2['mid_p50']

plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')



x4 = df2['mid_p75']

plt.scatter(x4, y, color='#2b5bde', label = '75th pct. Mid Career Salary')



x5 = df2['mid_p90']

plt.scatter(x5, y, color='#a1b6f0', label = '90th pct. Mid Career Salary')



plt.xlabel('US $')

plt.ylabel('')

plt.title('Salary Information by Major')

plt.legend(loc='upper right', bbox_to_anchor=(1.42,.98))



plt.grid(True) #turn grid on



plt.show()
df2 = df1.sort_values(by = 'mid_p50', ascending = False)

df2 = df2.reset_index()



fig = plt.figure(figsize=(8,12))

matplotlib.rc('grid', alpha = .5, color = '#e3dfdf')   #color the grid lines

matplotlib.rc('axes', edgecolor = '#67746A')           #color the graph edge

#matplotlib.rcParams.update(matplotlib.rcParamsDefault)  #this will reset default params if you need



x = df2['bgn_p50']

y = len(df2.index) - df2.index + 1

labels = df2['major']



#plt.scatter(x, y, color='b', label = 'Median Starting Salary')

plt.yticks(y, labels)



x1 = df2['mid_p10']

plt.scatter(x1, y, color='#f7e9ad', label = '10th pct. Mid Career Salary')



x2 = df2['mid_p25']

plt.scatter(x2, y, color='#ecc833', label = '25th pct. Mid Career Salary')



x3 = df2['mid_p50']

plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')



x4 = df2['mid_p75']

plt.scatter(x4, y, color='#2b5bde', label = '75th pct. Mid Career Salary')



x5 = df2['mid_p90']

plt.scatter(x5, y, color='#a1b6f0', label = '90th pct. Mid Career Salary')



# make clear the range of salary

for i in range(len(df2.index)):

    plt.plot([x1[i], x5[i]], [y[i], y[i]], color='gray')

plt.plot(x1, y, color='#f7e9ad')

plt.plot(x5, y, color='#a1b6f0')



plt.xlabel('US $')

plt.ylabel('')

plt.title('Salary Information by Major')

plt.legend(loc='upper right', bbox_to_anchor=(1.42,.98))



plt.grid(True) #turn grid on



plt.show()