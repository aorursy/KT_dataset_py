import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import math
for i in range(len(os.listdir('../input'))-1):

    year = str(2001+i)

    name = 'table' + year + '.csv'

    if i == 0:

        data = pd.read_csv('../input/'+name, encoding='shift-jis')

    else:

        data = pd.concat([data,pd.read_csv('../input/'+name, encoding='shift-jis')],axis=0)

data = data.rename(columns = {'年':'year','月':'month','日':'day','時（UTC）':'hour','台風番号':'typhoon number','台風名':'typhoon name','階級':'grade','緯度':'longitude','経度':'latitude','中心気圧':'center pressure','最大風速':'max wind speed','50KT長径方向':'50KT major axis direction','50KT長径':'50KT major axis diameter','50KT短径':'50KT minor axis diameter','30KT長径方向':'30KT major axis direction','30KT長径':'30KT major axis diameter','30KT短径':'30KT minor axis diameter','上陸':'landing'})
data.head()
for col in ['year','month','day','hour']:

    print(col)

    print(np.sort(data[col].unique()))
print(data['typhoon number'].unique().shape)

print(data['typhoon name'].unique().shape)
data['grade'].value_counts()
name = 'CIMARON'

num = 101

x = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['latitude'].values

y = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['longitude'].values

fig = plt.figure(facecolor="skyblue")

plt.plot(x,y,'darkorchid')

plt.hlines([10,20,30],110,140,"black", linestyles='dashed',linewidth=1)

plt.vlines([110,120,130,140],10,30,"black", linestyles='dashed',linewidth=1)

plt.show()
example_num = 1

jp_x = [145,143,141,142,140,136,133,131,129,133,137,139,140,141,142,145]

jp_y = [44,41,42,40,35,34,33,30,33,36,37,38,41,42,45,44]



for name in data['typhoon name'].unique()[:example_num]:

    print(name)

    for num in data[data['typhoon name']==name]['typhoon number'].unique():

        

        x = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['latitude'].values

        y = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['longitude'].values

        

        plt.figure(facecolor="white")

        plt.axhspan(10,50,0.04,0.96,color='skyblue')

        plt.fill(jp_x,jp_y,color="g",alpha=0.5)

        plt.plot(x,y,'darkorchid')

        plt.xlabel('longitude')

        plt.ylabel('latitude')

        plt.hlines([10,20,30,40,50],110,160,"black", linestyles='dashed',linewidth=1)

        plt.vlines([110,120,130,140,150,160],10,50,"black", linestyles='dashed',linewidth=1)

        plt.show()
print(np.sort(data['latitude'].values))

print(np.sort(data['longitude'].values))
np.sort(data['center pressure'].unique())
data['max wind speed'].unique()
example_num = 1

jp_x = [145,143,141,142,140,136,133,131,129,133,137,139,140,141,142,145]

jp_y = [44,41,42,40,35,34,33,30,33,36,37,38,41,42,45,44]



for name in data['typhoon name'].unique()[:example_num]:

    print(name)

    for num in data[data['typhoon name']==name]['typhoon number'].unique():

        print(num)

        x = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['latitude'].values

        y = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['longitude'].values

        

        major_diameter_50 = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['50KT major axis diameter'].values

        major_direction_50 = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['50KT major axis direction'].values

        minor_diameter_50 = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['50KT minor axis diameter'].values

        major_diameter_30 = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['30KT major axis diameter'].values

        major_direction_30 = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['30KT major axis direction'].values

        minor_diameter_30 = data[(data['typhoon name']==name)&(data['typhoon number']==num)]['30KT minor axis diameter'].values

        shifted_x = x + ((major_diameter_50-minor_diameter_50)/2)*np.sin((np.pi/4)*major_direction_50)*(1.852/111)

        shifted_y = y + ((major_diameter_50-minor_diameter_50)/2)*np.cos((np.pi/4)*major_direction_50)*(1.852/111)

        diameter = ((major_diameter_50+minor_diameter_50)/2)*(1.852/111)

        diameter_30 = ((major_diameter_30+minor_diameter_30)/2)*(1.852/111)

        fig = plt.figure(facecolor="white")

        plt.axhspan(10,50,0.04,0.96,color='skyblue')

        plt.fill(jp_x,jp_y,color="g",alpha=0.5)

        # 50 knot storm range

        for i in range(len(shifted_x)):

            circle_x = []

            circle_y = []

            for j in np.linspace(-180,180,360):

                circle_x.append(shifted_x[i]+diameter[i]*math.cos(math.radians(j)))

                circle_y.append(shifted_y[i]+diameter[i]*math.sin(math.radians(j)))

            plt.plot(circle_x,circle_y,'red')

        # 30 knot storm range

        for i in range(len(shifted_x)):

            circle_x = []

            circle_y = []

            for j in np.linspace(-180,180,360):

                circle_x.append(shifted_x[i]+diameter_30[i]*math.cos(math.radians(j)))

                circle_y.append(shifted_y[i]+diameter_30[i]*math.sin(math.radians(j)))

            plt.plot(circle_x,circle_y,'green',linewidth=0.5)

        plt.plot(x,y)

        plt.plot(jp_x,jp_y,'black')

        plt.plot(shifted_x,shifted_y,'red')

        plt.xlabel('longitude')

        plt.ylabel('latitude')

        plt.hlines([10,20,30,40,50],110,160,"black", linestyles='dashed',linewidth=1)

        plt.vlines([110,120,130,140,150,160],10,50,"black", linestyles='dashed',linewidth=1)

        plt.show()
print('Showing numbers of typhoons for each year')

numbers = {}

for year in data['year'].unique():

    numbers[year] = len(data[data['year']==year]['typhoon number'].unique())

    #print(str(year)+' : ',len(data[data['year']==year]['typhoon number'].unique()))

plt.figure(facecolor='gray')

plt.plot(list(numbers.keys()),list(numbers.values()),'black')

plt.show()

print('correlation coefficient : '+str(np.corrcoef(list(numbers.keys()),list(numbers.values()))[0][1]))
numbers = {}

for year in data['year'].unique()[9:]:

    numbers[year] = len(data[data['year']==year]['typhoon number'].unique())

    #print(str(year)+' : ',len(data[data['year']==year]['typhoon number'].unique()))

plt.figure(facecolor='gray')

plt.plot(list(numbers.keys()),list(numbers.values()))

plt.show()

print('correlation coefficient : '+str(np.corrcoef(list(numbers.keys()),list(numbers.values()))[0][1]))
print('Showing average size of typhoons for each year')

print('Circle size reveals typhoon size')

sizes = {}

for year in data['year'].unique():

    sizes[year] = np.mean((data[data['year']==year]['50KT major axis diameter'].values + data[data['year']==year]['50KT minor axis diameter'].values)/2)

plt.figure(facecolor='gray')

plt.plot(list(sizes.keys()),list(sizes.values()),'black')

for i in range(len(sizes)):

    circle_x = []

    circle_y = []

    for j in np.linspace(-180,180,360):

        circle_x.append(list(sizes.keys())[i]+(list(sizes.values())[i]/50)*math.cos(math.radians(j)))

        circle_y.append(list(sizes.values())[i]+(list(sizes.values())[i]/30)*math.sin(math.radians(j)))

    plt.plot(circle_x,circle_y)

plt.show()

print('correlation coefficient : '+str(np.corrcoef(list(sizes.keys()),list(sizes.values()))[0][1]))
print('Showing max size of typhoons for each year')

sizes = {}

for year in data['year'].unique():

    sizes[year] = max((data[data['year']==year]['50KT major axis diameter'].values + data[data['year']==year]['50KT minor axis diameter'].values)/2)

plt.figure(facecolor='gray')

for i in range(len(sizes)):

    circle_x = []

    circle_y = []

    for j in np.linspace(-180,180,360):

        circle_x.append(list(sizes.keys())[i]+(list(sizes.values())[i]/300)*math.cos(math.radians(j)))

        circle_y.append(list(sizes.values())[i]+(list(sizes.values())[i]/40)*math.sin(math.radians(j)))

    plt.plot(circle_x,circle_y)

plt.plot(list(sizes.keys()),list(sizes.values()))

plt.show()

print('correlation coefficient : '+str(np.corrcoef(list(sizes.keys()),list(sizes.values()))[0][1]))
print('Showing average max wind speed of typhoons for each year')

sizes = {}

for year in data['year'].unique():

    sizes[year] = np.mean(data[data['year']==year]['max wind speed'].values)

plt.figure(facecolor='gray')

plt.plot(list(sizes.keys()),list(sizes.values()),'black')

plt.show()

print('correlation coefficient : '+str(np.corrcoef(list(sizes.keys()),list(sizes.values()))[0][1]))
print('Showing numbers of landed typhoons for each year')

numbers = {}

for year in data['year'].unique():

    count = 0

    for num in data[data['year']==year]['typhoon number'].unique():

        if not (len(data[(data['year']==year)&(data['typhoon number']==num)]['landing'].unique())==1)&(data[(data['year']==year)&(data['typhoon number']==num)]['landing'].unique()[0]==0):

            count += 1

    numbers[year] = count

plt.figure(facecolor='gray')

plt.plot(list(numbers.keys()),list(numbers.values()))

plt.show()

print('correlation coefficient : '+str(np.corrcoef(list(numbers.keys()),list(numbers.values()))[0][1]))