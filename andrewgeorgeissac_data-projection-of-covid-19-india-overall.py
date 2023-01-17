#import

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

#importing dataset

dataset=pd.read_csv('../input/corona-virus-details-of-india/daybydayanalysis.csv').values

#segregating data

dates=dataset[:,0] #dates

cases=dataset[:,1] #cases

deaths=dataset[:,2] #deaths

recovered=dataset[:,3] #recovered

#COVID-19 cases projection

plt.figure(figsize=(30,10))

plt.bar(dates,cases,width=0.3,color='red')

plt.plot(dates,cases,color='black')

plt.show()

#COVID-19 cases and recovery 

plt.figure(figsize=(30,10))

plt.title('Cases Vs Recovery')

plt.plot(dates,cases,color='red')

plt.plot(dates,recovered,color='green')

plt.show()

index=np.arange(len(dates))

width=0.30

#bar graph

plt.figure(figsize=(30,10))

plt.title('Cases Vs Recovery')

plt.bar(index,cases,color='red',label='Cases',width=0.30)

plt.bar(index+width,recovered,color='green',label='Recovered',width=0.30)

plt.xticks(index+width/2,dates)

plt.xlabel('Dates')

plt.ylabel('Values')

plt.legend()

plt.show()

#COVID-19 recovery & deaths

plt.figure(figsize=(30,10))

plt.plot(dates,recovered,color='green')

plt.plot(dates,deaths,color='black')

plt.show()

index=np.arange(len(dates))

width=0.30

#bar graph

plt.figure(figsize=(30,10))

plt.bar(index,recovered,color='green',label='Recovered',width=0.30)

plt.bar(index+width,deaths,color='black',label='Deaths',width=0.30)

plt.xticks(index+width/2,dates)

plt.xlabel('Dates')

plt.ylabel('Values')

plt.legend()

plt.show()