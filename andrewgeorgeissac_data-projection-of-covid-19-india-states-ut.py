import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import numpy as np
dataset=pd.read_csv('../input/corona-virus-details-of-india/coronacases28042020.csv') #importing the data set

states=dataset.iloc[:,0] #names of states & union territories

#NOTE: ALL STATES/UNION TERRITORIES ARE SHOWN IN SHORTFORMS

confirmed=dataset.iloc[:,1] #confirmed cases

deaths=dataset.iloc[:,2] #deaths

recovered=dataset.iloc[:,3] #recovered

#confirmed cases

plt.figure(figsize=(30,10))

plt.bar(states,confirmed,color='red')

plt.show()

plt.figure(figsize=(30,10))

plt.pie(confirmed,labels=states)

plt.show()
#deaths

plt.figure(figsize=(30,10))

plt.bar(states,deaths,color='black')

plt.show()
#recovered

plt.figure(figsize=(30,10))

plt.bar(states,recovered,color='green')

plt.show()