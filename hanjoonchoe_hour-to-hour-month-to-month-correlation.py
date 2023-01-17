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



import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

import math

from scipy.spatial import distance

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')
train.head()

train.columns
train.groupby('Hour').TotalTimeStopped_p80.sum().plot(kind='bar')
train['Intersection'] = train.IntersectionId.astype(str) + train.City

lists=train.groupby(['Intersection']).apply(lambda x : x.sort_values(['TotalTimeStopped_p80'])).reset_index(drop=True)

lists_T = lists[['Intersection','Hour','TotalTimeStopped_p80']]

lists_T.isnull().sum()
def T1_TNplot(df,time):

    plustime = np.linspace(1,23-time,23-time).astype(int)

    fig = plt.figure(figsize=(25, 16))

    for i in plustime:

        size = df.groupby('Hour').TotalTimeStopped_p80.count()

        less = min(size[time], size[(time+i)%24]) 

        timeN = df.loc[df.Hour == time].iloc[:less]

        timeT = df.loc[df.Hour==(time+i)%24].iloc[:less]

        idx = timeN.Intersection.values == timeT.Intersection.values

        concatTime = pd.DataFrame({f'{time}':timeN.TotalTimeStopped_p80.loc[idx].values

                                   ,f"{time}+{i}":timeT.TotalTimeStopped_p80.loc[idx].values})

        ax = fig.add_subplot(6, 4, i)#, xticks=[], yticks=[])

        sns.regplot(x=f'{time}',y=f"{time}+{i}",data=concatTime)

        ax.set_title(f'{time}+{i}_Hour')

    plt.show()



for j in range(24):

    print(j,'Hour')

    T1_TNplot(lists_T,j)
lists_M = lists[['Intersection','Month','TotalTimeStopped_p80']]

lists_M.isnull().sum()
def MT_MNplot(df,month,month_size):

    size = df.groupby('Month').TotalTimeStopped_p80.count()

    fig = plt.figure(figsize=(25, 16))

    for i in month_size:

        less = min(size[month], size[i]) 

        monthN = df.loc[df.Month == month].iloc[:less]

        monthT = df.loc[df.Month==i].iloc[:less]

        idx = monthN.Intersection.values == monthT.Intersection.values

        concatTime = pd.DataFrame({f'{month}':monthN.TotalTimeStopped_p80.loc[idx].values

                                   ,f"{i}":monthT.TotalTimeStopped_p80.loc[idx].values})

        ax = fig.add_subplot(6, 4, i, xticks=[], yticks=[])

        sns.regplot(x=f'{month}',y=f"{i}",data=concatTime)

        ax.set_title(f'{month} vs {i}_month')

    plt.show()

month_size = np.sort(lists_M.Month.unique(), axis=0)

for j in month_size:

    MT_MNplot(lists_M,j,month_size)