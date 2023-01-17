# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib # data visualization

import matplotlib.pyplot as plt



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv('../input/challenge.csv')
def toTime(a):

    hr, mi, se = map(int, a.split(':'))

    # Returning time in seconds

    return (hr*3600) + (mi * 60) + se



df = df.dropna()



total_marathon_length = 42.195

half_way_length = total_marathon_length / 2

# Distributing the marathon into 4 legs and calculating length in meters for each leg

FirstPart = 10 * 1000

SecondPart = ((total_marathon_length / 2) - 10) * 1000

ThirdPart = ( 30 - (total_marathon_length / 2)) * 1000

FinalPart = (total_marathon_length - 30) * 1000



# Calculating the time in second for each leg

df['10KmSec'] = df['10km Time'].apply(toTime)

df['HalfWaySec'] = df['Half Way Time'].apply(toTime)

df['30KmSec'] = df['30km Time'].apply(toTime)

df['NetSec'] = df['Net Time'].apply(toTime)



#Calculating average speeds for each contestant in each of the legs

df['1st Leg'] = FirstPart / df['10KmSec']

df['2nd Leg'] = SecondPart / (df['HalfWaySec'] - df['10KmSec'])

df['3rd Leg'] = ThirdPart / (df['30KmSec'] - df['HalfWaySec'])

df['4th Leg'] = FinalPart / (df['NetSec'] - df['30KmSec'])



cols = ['1st Leg', '2nd Leg', '3rd Leg', '4th Leg']

df_reduced = df[['Overall Position'] + cols]



avg = df_reduced[cols].mean()

top20 = df_reduced[df_reduced['Overall Position'] < 21][cols].mean()



# List to store average time for various categories

Series_list = [top20]

# List to store names of the various categories

ColumnList = ['Top20']



max_overall = df_reduced['Overall Position'].max()

i = 121

while i < max_overall:

    Series_list.append(df_reduced[ df_reduced['Overall Position'] < i ][cols].mean())

    ColumnList.append( 'Till Rank : ' + str(i) )

    i += 300

Series_list.append(avg)

ColumnList.append('Overall Avg')

df_eval = pd.concat(Series_list, axis = 1)

df_eval.columns = ColumnList

df_eval = df_eval.transpose()
font = {'size'   : 32}

matplotlib.rc('font', **font)

ax = df_eval.plot(kind = 'line', figsize = (16, 8), title = 'Comparison of average speed for different legs of the race'

             , fontsize =  10)

ax.set_ylabel("Average Speed")

ax.set_xlabel("Ranks till which average was calculated")
df_reduced = df[['Country '] + cols]

Series_list = []

ColumnList = []

countries = df['Country '].unique()

for country in countries:

    Series_list.append(df_reduced[ df_reduced['Country '] == country][cols].mean())

    ColumnList.append(country)

df_eval = pd.concat(Series_list, axis = 1)

df_eval.columns = ColumnList

df_eval = df_eval.transpose()

df_eval['mean'] = df_eval.mean(axis = 1)

df_eval.sort_values(by='mean', ascending = False, inplace = True)

df_eval = df_eval.reset_index()

df_top10 = df_eval.loc[:9]

series_others = pd.Series({'index':'Others'})

series_others = series_others.append(df_eval.loc[10:,cols + ['mean']].mean())

df_top10 = df_top10.append(series_others, ignore_index = True)

df_top10 = df_top10.transpose()

df_top10.columns = df_top10.loc['index']

df_top10 = df_top10[df_top10.index != 'index']
font = {'size'   : 32}

matplotlib.rc('font', **font)

ax = df_top10.loc['1st Leg':'4th Leg'].plot(kind = 'line', figsize = (16, 8),

                                            title = 'Comparison of average speed for different legs of the race'

                                             , fontsize =  10, colormap = 'Accent')

ax.set_ylabel("Average Speed")

ax.set_xlabel("Legs of the race")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)