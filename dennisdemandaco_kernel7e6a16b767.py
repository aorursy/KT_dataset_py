# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
# Code below for combining multiple csv file of Divvy Bikes
import os

import glob

import pandas as pd



def contatenate(indir = "C:\\Users\\WNJK84\\Desktop\\Divvy", outfile="C:\\Users\\WNJK84\\Desktop\\Divvy\\dd.csv"):

    os.chdir(indir)

    fileList = glob.glob("*.csv")

    dfList = []

    colnames = ['trip_id', 'starttime', 'stoptime', 'bikeid', 'tripduration', 'from_station_id', 'from_station_name', 'to_station_id', 'to_station_name', 'usertype', 'gender', 'birthday']

    for filename in fileList:

        print(filename)

        df = pd.read_csv(filename, header = None)

        dfList.append(df)

    concatDf = pd.concat(dfList, axis = 0)

    concatDf.columns = colnames

    concatDf.to_csv(outfile, index = None)
# Read Combined CSV

data = pd.read_csv('Divvyall.csv')

data
# Top 10 Start Locations for Divvy Bikes Usage

top10_start = data.pivot_table(index = ['from_station_name'],  aggfunc='size').sort_values(ascending = False).head(10)

starts = top10_start.index

print(top10_start)



# from_station_name

# Streeter Dr & Illinois St       

# Lake Shore Dr & Monroe St       

# Millennium Park                 

# Clinton St & Washington Blvd    

# Michigan Ave & Oak St           

# Theater on the Lake             

# Museum Campus                   

# McClurg Ct & Illinois St        

# Canal St & Madison St           

# Michigan Ave & Lake St          

# dtype: int64