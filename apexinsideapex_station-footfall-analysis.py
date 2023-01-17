# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading Data



d1 = pd.read_csv("/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q1).csv")

d2 = pd.read_csv("/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q2).csv")

d3 = pd.read_csv("/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q3).csv")

d4 = pd.read_csv("/kaggle/input/toronto-bikeshare-data/bikeshare-ridership-2017/2017 Data/Bikeshare Ridership (2017 Q4).csv")

d5 = pd.read_csv("/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv")

d6 = pd.read_csv("/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv")

d7 = pd.read_csv("/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv")

d8 = pd.read_csv("/kaggle/input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv")



#Making things easier

data = d1.copy()

data = data.append([d2,d3,d4,d5,d6,d7,d8], sort=True)
#Cleaning the data

data = data.dropna()

num_occurances = dict(Counter(data.from_station_name))



top_trip_starting_area = sorted(num_occurances.items(), key = lambda x : x[1], reverse=True)[:5]

top_trip_ending_area = sorted(dict(Counter(data.to_station_name)).items(), key= lambda x : x[1], reverse= True)[:5]
#Results



print("The top 5 stations to start from are: \n")

print(*top_trip_starting_area, sep="\n")

print()

print("The top 5 destinations are: \n")

print(*top_trip_ending_area, sep="\n")
