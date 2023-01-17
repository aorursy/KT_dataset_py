# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/nycweather/nyc_Jan_Jun_2016_weat.csv")#pd.read_csv basically read the csv data and we store in variable data

                                                                   #called as dataframe(the most powerful data structure of pandas just like multidimensional array is most powerful datastructure of numpy)

data


print(data['LATITUDE'].max())

data['LATITUDE'].min()
print(data['LATITUDE'][data['TMAX']>=82])#printing latitude where tmax>=82

print(data['LATITUDE'].mean(),data['LATITUDE'].median(),data['LATITUDE'].var())#print mean,median of latitude
data.shape
data.tail#for printing some of buttom row
data.head #for printing some of data from top
print(data[2:5],end='\n\n') #printing data by slicing

print(data['LATITUDE'],end='\n\n')#print particular columnn data ,,,,u can also write as data.LATITUDE

print(data.columns)#print all the feature/column
print(data[['LATITUDE','SNOW']][2:4],end='\n\n') #print 2column simultaneously for 2nd and 3rd index

print(data[['LATITUDE','SNOW','TAVG']]) #print 3column simultaneously
print(data[['LATITUDE','SNOW']].min())

print(data[['LATITUDE','SNOW']].describe())#describe give all information like min,max,standard deviation etc.,You can use describe for one or more element simultaneously
print(data[data['SNOW']==data.SNOW.max()],end='\n\n') #selecting row whose snow is maximum

print(data[['LATITUDE','SNOW']][data['SNOW']==data.SNOW.max()],end='\n\n') #selecting latitude and snow whose snow is maximum

print(data.LATITUDE[data.SNOW==data.SNOW.max()],end='\n\n')#another way to write previous statement using dot as accessing 

data['SNOW'][data['LATITUDE']>=41.082]#selecting data of snow whose latitude is more than 41.082