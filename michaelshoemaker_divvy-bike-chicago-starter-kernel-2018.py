# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv(r'../input/Divvy_Trips_2018_Q1.csv',parse_dates=['start_time','end_time'])
df2 = pd.read_csv(r'../input/Divvy_Trips_2018_Q2.csv',parse_dates=['start_time','end_time'])
df3 = pd.read_csv(r'../input/Divvy_Trips_2018_Q3.csv',parse_dates=['start_time','end_time'])
df4 = pd.read_csv(r'../input/Divvy_Trips_2018_Q4.csv',parse_dates=['start_time','end_time'])
frames = [df1, df2, df3, df4]
df = pd.concat(frames)
print('The sum of the rows in the combined dataframe is {} and the sum of the rows in the individual '
      'dataframes is {}'.format(df.shape[0],df1.shape[0]+df2.shape[0]+df3.shape[0]+df4.shape[0]))