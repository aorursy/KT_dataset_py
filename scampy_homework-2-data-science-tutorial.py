# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Import Data
df=pd.read_csv('../input/fifa_ranking.csv')
#Summary of Data
df.info()
#Basic statistics of data
df.describe()
#First 5 rows of the data
df.head(5)
df.rank_date.unique()
# Filter data with country name Turkey and assign it to a new DataFrame
rank_date_filter = df.country_full == 'Turkey'
df_Turkey = df[rank_date_filter].copy().reset_index(drop=True)
df_Turkey.head(5)
#Drop the total_point
point_filter = df_Turkey['total_points'] != 0
del_index = df_Turkey[point_filter].index[0]
df_Turkey.drop(df_Turkey.index[:del_index],inplace=True)
#Reset Index
df_Turkey= df_Turkey.reset_index(drop=True)
df_Turkey.head(5)
#Define average function and Create new Column assign High-Average-Low string according to funcntion
def take_average():
    average = sum(df_Turkey.total_points)/len(df_Turkey.total_points)
    df_Turkey['point_scale'] = ['High' if (point > average) else 'Average' if (average <= point < (average + 100)) else 'Low' for point in df_Turkey.total_points]
    return average
average = take_average()
#Check total_points vs point_scale
df_Turkey.loc[:10,['total_points','point_scale']]
plt.figure(figsize=(18, 18))
plt.scatter(df_Turkey.rank_date,df_Turkey.total_points,color = 'r', label = 'Point')
plt.axhline(average, color='g', linestyle='dashed', linewidth=1,label = 'Average Point')
plt.xlabel('Rank Date')
plt.xticks(rotation='vertical')
plt.ylabel('Total Points')
plt.title('Turkey Total Points vs Date')
plt.legend()
plt.show()