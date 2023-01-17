# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/results.csv', names = ['Gender', 'Event', 'Location', 'Year', 'Medal', 'Name', 'Nationality', 'Result', 'Wind'])
def exponenial_func(x, a, b, c):

    return a*np.exp(-b*x)+c
sprint = df[df.Event == '100M Men']

sprint = sprint[sprint.Result != 'None']

sprint['Year'] = sprint.Year.apply(lambda x: int(x))

sprint['Result'] = sprint.Result.apply(lambda x: float(x))



plt.figure(figsize = (8,8))

plt.plot(sprint.Year[sprint.Medal == 'G'].values,sprint.Result[sprint.Medal == 'G'].values,'o', color = 'y', markersize = 10, alpha = 0.4)

plt.plot(sprint.Year[sprint.Medal == 'S'].values,sprint.Result[sprint.Medal == 'S'].values,'o', color = 'gray',markersize = 10, alpha = 0.4)

plt.plot(sprint.Year[sprint.Medal == 'B'].values,sprint.Result[sprint.Medal == 'B'].values,'o', color = 'r',markersize = 10, alpha = 0.4)



x = sprint.Year.values - 1896

y = sprint.Result

popt, pcov = curve_fit(exponenial_func, x, y, p0=(12, 1e-6, 1))

yy = exponenial_func(np.unique(x), *popt)

plt.plot(np.unique(sprint.Year.values),yy)

plt.xlabel('Year')

plt.ylabel('Result [sec]')

plt.legend(['Gold','Silver','Bronze'])

plt.title('100M Men')
sprint = df[df.Event == '100M Women']

sprint = sprint[sprint.Result != 'None']

sprint['Year'] = sprint.Year.apply(lambda x: int(x))

sprint['Result'] = sprint.Result.apply(lambda x: float(x))



plt.figure(figsize = (8,8))

plt.plot(sprint.Year[sprint.Medal == 'G'].values,sprint.Result[sprint.Medal == 'G'].values,'o', color = 'y', markersize = 10, alpha = 0.4)

plt.plot(sprint.Year[sprint.Medal == 'S'].values,sprint.Result[sprint.Medal == 'S'].values,'o', color = 'gray',markersize = 10, alpha = 0.4)

plt.plot(sprint.Year[sprint.Medal == 'B'].values,sprint.Result[sprint.Medal == 'B'].values,'o', color = 'r',markersize = 10, alpha = 0.4)







x = sprint.Year.values - 1932

y = sprint.Result

popt, pcov = curve_fit(exponenial_func, x, y, p0=(12, 1e-6, 1))

yy = exponenial_func(np.unique(x), *popt)

plt.plot(np.unique(sprint.Year.values),yy, 'b')





popt, pcov = curve_fit(exponenial_func, x, y, p0=(12, 1e-6, 1))

yy = exponenial_func(np.unique(x), *popt)

plt.plot(np.unique(sprint.Year.values),yy)

plt.xlabel('Year')

plt.ylabel('Result [sec]')

plt.legend(['Gold','Silver','Bronze'])

plt.title('100M Women')
sprint = df[df.Event == '200M Men']

sprint = sprint[sprint.Result != 'None']

sprint['Year'] = sprint.Year.apply(lambda x: int(x))

sprint['Result'] = sprint.Result.apply(lambda x: float(x))



plt.figure(figsize = (8,8))

plt.plot(sprint.Year[sprint.Medal == 'G'].values,sprint.Result[sprint.Medal == 'G'].values,'o', color = 'y', markersize = 10, alpha = 0.4)

plt.plot(sprint.Year[sprint.Medal == 'S'].values,sprint.Result[sprint.Medal == 'S'].values,'o', color = 'gray',markersize = 10, alpha = 0.4)

plt.plot(sprint.Year[sprint.Medal == 'B'].values,sprint.Result[sprint.Medal == 'B'].values,'o', color = 'r',markersize = 10, alpha = 0.4)



x = sprint.Year.values - 1896

y = sprint.Result

popt, pcov = curve_fit(exponenial_func, x, y, p0=(12, 1e-6, 1))

yy = exponenial_func(np.unique(x), *popt)

plt.plot(np.unique(sprint.Year.values),yy)

plt.xlabel('Year')

plt.ylabel('Result [sec]')

plt.legend(['Gold','Silver','Bronze'])

plt.title('200M Men')
sprint = df[df.Event == '200M Women']

sprint = sprint[sprint.Result != 'None']

sprint['Year'] = sprint.Year.apply(lambda x: int(x))

sprint['Result'] = sprint.Result.apply(lambda x: float(x))



plt.figure(figsize = (8,8))

plt.plot(sprint.Year[sprint.Medal == 'G'].values,sprint.Result[sprint.Medal == 'G'].values,'o', color = 'y', markersize = 10, alpha = 0.4)

plt.plot(sprint.Year[sprint.Medal == 'S'].values,sprint.Result[sprint.Medal == 'S'].values,'o', color = 'gray',markersize = 10, alpha = 0.4)

plt.plot(sprint.Year[sprint.Medal == 'B'].values,sprint.Result[sprint.Medal == 'B'].values,'o', color = 'r',markersize = 10, alpha = 0.4)



x = sprint.Year.values - 1948

y = sprint.Result

popt, pcov = curve_fit(exponenial_func, x, y, p0=(12, 1e-6, 1))

yy = exponenial_func(np.unique(x), *popt)

plt.plot(np.unique(sprint.Year.values),yy)

plt.xlabel('Year')

plt.ylabel('Result [sec]')

plt.legend(['Gold','Silver','Bronze'])

plt.title('200M Women')
def get_expectation(df,event,year):

    temp_df = df[df.Event == event]

    temp_df = temp_df[temp_df.Result != 'None']

    temp_df['Year'] = temp_df.Year.apply(lambda x: int(x))

    temp_df['Result'] = temp_df.Result.apply(lambda x: float(x))



    x = temp_df.Year.values - temp_df.Year.min()

    y = temp_df.Result

    popt, pcov = curve_fit(exponenial_func, x, y, p0=(np.mean(y), 1e-6, 1))

    return exponenial_func(year -  temp_df.Year.min() , *popt)





get_expectation(df,'200M Men',2016)
sprint = df[df.Event == '100M Men']

sprint = sprint[sprint.Result != 'None']

sprint['Year'] = sprint.Year.apply(lambda x: int(x))

sprint['Result'] = sprint.Result.apply(lambda x: float(x))

sprint = sprint.reset_index()

sprint['expectation'] = 0

for i,row in enumerate(sprint.iterrows()):

    sprint.loc[i,'expectation'] = get_expectation(df,'100M Men',float(sprint.loc[i,'Year']))

    

sprint['ratio'] = np.true_divide(sprint.expectation,sprint.Result)

sprint.sort_values('ratio', ascending = False).head(20)[['Name','Year','Result','expectation']].reset_index().drop('index', axis = 1)
sprint = df[df.Event == '100M Women']

sprint = sprint[sprint.Result != 'None']

sprint['Year'] = sprint.Year.apply(lambda x: int(x))

sprint['Result'] = sprint.Result.apply(lambda x: float(x))

sprint = sprint.reset_index()

sprint['expectation'] = 0

for i,row in enumerate(sprint.iterrows()):

    sprint.loc[i,'expectation'] = get_expectation(df,'100M Women',float(sprint.loc[i,'Year']))

    

sprint['ratio'] = np.true_divide(sprint.expectation,sprint.Result)

sprint.sort_values('ratio', ascending = False).head(20)[['Name','Year','Result','expectation']].reset_index().drop('index', axis = 1)
sprint = df[df.Event == '200M Men']

sprint = sprint[sprint.Result != 'None']

sprint['Year'] = sprint.Year.apply(lambda x: int(x))

sprint['Result'] = sprint.Result.apply(lambda x: float(x))

sprint = sprint.reset_index()

sprint['expectation'] = 0

for i,row in enumerate(sprint.iterrows()):

    sprint.loc[i,'expectation'] = get_expectation(df,'200M Men',float(sprint.loc[i,'Year']))

    

sprint['ratio'] = np.true_divide(sprint.expectation,sprint.Result)

sprint.sort_values('ratio', ascending = False).head(20)[['Name','Year','Result','expectation']].reset_index().drop('index', axis = 1)
sprint = df[df.Event == '200M Women']

sprint = sprint[sprint.Result != 'None']

sprint['Year'] = sprint.Year.apply(lambda x: int(x))

sprint['Result'] = sprint.Result.apply(lambda x: float(x))

sprint = sprint.reset_index()

sprint['expectation'] = 0

for i,row in enumerate(sprint.iterrows()):

    sprint.loc[i,'expectation'] = get_expectation(df,'200M Women',float(sprint.loc[i,'Year']))

    

sprint['ratio'] = np.true_divide(sprint.expectation,sprint.Result)

sprint.sort_values('ratio', ascending = False).head(20)[['Name','Year','Result','expectation']].reset_index().drop('index', axis = 1)