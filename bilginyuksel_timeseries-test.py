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

const_fpath = '/kaggle/input/temperature-timeseries-for-some-brazilian-cities/'
sample = pd.read_csv('/kaggle/input/temperature-timeseries-for-some-brazilian-cities/station_curitiba.csv')

sample.sample(5)
from datetime import datetime



class Cleaner:

    def __init__(self, data):

        self.data = data

    

    def drop_columns(self):

        columns_should_remove = ['D-J-F', 'M-A-M', 'J-J-A', 'S-O-N', 'metANN']

        self.data = self.data.drop(columns_should_remove, axis=1)

        return self

    

    def manage_timeseries(self):

        timeseries_data = []

        

        # Maybe I can use a simple pandas function here but I don't know what to use so I will basically use programming to

        # solve this issue.

        unique_years = self.data.YEAR.unique()

        col = list(self.data.columns[1:])

        timeseries_dictionary = {} # keys will be the times and values will be the temperatures

        for i in unique_years:

            for j in range(1, 13): # We know that every year have 12 months

                timeseries_data.append([datetime(i, j, 1), float(self.data[self.data.YEAR==i][col[j-1]])])

        

        self.data = pd.DataFrame(timeseries_data, columns=['timeseries', 'temperature'])    

        return self

    

    def add_city(self, state):

        self.data['state'] = state

        return self

    

    def build(self):

        return self.data

        

# data = pd.read_csv('/kaggle/input/temperature-timeseries-for-some-brazilian-cities/station_belem.csv')



# cleaner = Cleaner(data)

# data = cleaner.drop_columns().manage_timeseries().add_city('belem').build()

# data.sample(5)
### THIS CODE READS ALL FILES AND CREATES ONE BIG DATASET BUT IT SEEMS LIKE NON SENSE



## Let's store all file's names

# files = []

# df = None

# dataset_list = []

# for dirname, _, filenames in os.walk('/kaggle/input/temperature-timeseries-for-some-brazilian-cities/'):

#     for filename in filenames:

#         data = pd.read_csv(os.path.join(dirname, filename))

#         city = filename.split('.')[0].split('_')[1]

#         dataset_list.append(data)

#         if df is None:

#             df = Cleaner(data).drop_columns().manage_timeseries().add_city(city).build()

#         else: pd.concat([df, Cleaner(data).drop_columns().manage_timeseries().add_city(city).build()])

        

# df.sample(20)
rio_data = pd.read_csv("".join([const_fpath, 'station_rio.csv']))

rio_data = Cleaner(rio_data).drop_columns().manage_timeseries().add_city('Rio').build()

rio_data.sample(5)
rio_data.info()
rio_data.describe()
# rio_data[rio_data.temperature==999.9] = rio_data.temperature.mean() # I just changed the values with the mean temperature not NaN

rio_data.loc[(rio_data.temperature == 999.9),'temperature']= rio_data.median()

rio_data.fillna(rio_data.median(), inplace=True)



import matplotlib.pyplot as plt

# rio_data.drop('state', axis = 1).plot()

plt.figure(figsize=(20, 5))

plt.title("Temperature by years")

plt.plot(rio_data.timeseries, rio_data.temperature)

plt.show()