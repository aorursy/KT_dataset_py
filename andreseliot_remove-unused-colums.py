import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import sys

from collections import defaultdict

import random
sys.version
def df_load(file, batch_size, skip = 1, usecolsCustom = None):

    ''' Build dataframe by iterating over chunks. Option to skip chunks and

        therefore read in less data. '''



    reader = pd.read_csv(file, nrows=batch_size,

                         dtype=np.float32, usecols=usecolsCustom)



    '''df = pd.concat((chunk for i, chunk in enumerate(reader) if i % skip == 0))'''



    return reader
input_path = '../input/'

df_num = df_load(input_path+'train_numeric.csv',

                 batch_size=100000, skip=20, usecolsCustom=['Id', 'Response'])
df_date = df_load(input_path+'train_date.csv',

                 batch_size=100000, skip=20)
df_date.head()
responsesTemp = []

for index, row in df_num.iterrows():

    if bool(row['Response']):

        responsesTemp.append(int(row['Id']))

responses = set(responsesTemp)
stations = defaultdict(list)

stationsdefaults = defaultdict(list)

for index, row in df_date.iterrows():

    current_station = ''

    for column in df_date:

        if(column != 'Id'):

            if (current_station != column.split('_')[1]):

                current_station = column.split('_')[1]

                stations[current_station]

                if not np.isnan(row[column]):

                    if (row['Id'] in responses):

                        stationsdefaults[current_station].append(row[column])

                    else:

                        stations[current_station].append(row[column])  
for j in range(51):

    values = [(random.random()*0.5 + 0.5) for i in range(len(stationsdefaults['S' + str(j)]))]

    plt.plot(stationsdefaults['S' + str(j)], values, marker='*', ls='', markersize='1')



    values = [random.random()*0.5 for i in range(len(stations['S' + str(j)]))]

    plt.plot(stations['S' + str(j)], values, marker='*', ls='', markersize='1')

    plt.title('Station S' + str(j))

    

    plt.show()

   