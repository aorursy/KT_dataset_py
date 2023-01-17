from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(rc={'figure.figsize':(10,10)}) 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw = pd.read_csv("/kaggle/input/buildingdatagenomeproject2/water_cleaned.csv", index_col = "timestamp", parse_dates = True)

raw.head()
raw.shape
import missingno as msno 

  

msno.matrix(raw) 

#lots of missing values, there seem to be two rows that are completely empty
raw.describe()
raw.columns
raw.apply('nunique') #num unique values for each column
raw.shape
print(raw.isnull().sum())
clean = raw.copy()
#not using this code, and unsure if it works properly

'''

import statistics

arr = []



for col in clean.columns:

    for i in clean.index:

        if pd.isnull(clean.loc[i, col]):

            if len(arr)>72:

                before = arr[len(arr): -72 :-1]

                avg = statistics.mean(before)

                arr.append(avg)

            else:

                arr.append(0)

        else:

            arr.append(clean.loc[i, col])

    clean = clean.drop(col, axis= 1)#drop column before adding new one with averages

    clean[col] = arr

    arr = [] #empty arr to restart a new array for next column

    

#clean.head()

'''
#clean.head()
msno.matrix(clean)
clean = clean.dropna(axis = 1)
clean.shape
day = raw.copy()

day = day.resample("D").mean()

day.shape
day.iloc[:200, :20].plot()
week= raw.copy()

week = week.resample("W").mean()

week.iloc[:200, :20].plot()
month= raw.copy()

month = month.resample("W").mean()

month.iloc[:200, :20].plot()
#percentage amount all the number of zeros per row

def zeros_per(df):

    for col in df.columns:

        count = 0

        for row in df.index:

            if df.loc[row, col] == 0.0:

                count += 1

        per = (count/(df.shape[0])) * 100

        print("{} | {}".format(

        df[col].name, 

        per

        ))
zeros_per(raw)
#dropping columns that have more than 75% of values as 0

clean = raw.copy()



for col in clean.columns:

    count = 0

    for row in clean.index:

        if clean.loc[row, col] == 0.0:

                count += 1

        per = (count/(clean.shape[0])) * 100

    if per > 75.0:

        clean = clean.drop(col, axis = 1)
clean.shape
#function shows the percentage of missing values and type of the values

def missing_data(data):

    percent = (data.isnull().sum() / data.isnull().count())

    x = pd.concat([percent], axis=1, keys=['Percentage_of_Missing_Values'])

    type = []

    

    for col in data.columns:

        dtype = str(data[col].dtype)

        type.append(dtype)

    x['Data Type'] = type

    

    return(np.transpose(x))
missing = missing_data(clean)

missing
#removing columns that have more than 50% missing values

for col in clean.columns:

    if missing.loc["Percentage_of_Missing_Values", col] >= .5:

        clean = clean.drop(col, axis = 1)
clean.shape
msno.matrix(clean)
clean = clean.interpolate(method='slinear')
msno.matrix(clean)
#few missing values

#back propagation fill

clean = clean.fillna(method='bfill')



#forward propagation fill 

clean = clean.fillna(method='ffill') 
msno.matrix(clean)
clean.to_csv("water_cleaned.csv")