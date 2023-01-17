# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%pylab



#import from course

%matplotlib inline

import math

import scipy.stats as stats

import matplotlib.pyplot as plt

#end import from course



from collections import defaultdict

from scipy.stats.stats import pearsonr

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#sample code

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

#end sample code



# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/HR_comma_sep.csv") 

#df.head()

#df.sales.head()

level=df['satisfaction_level'].values

time=df['last_evaluation'].values

projects=df['number_project'].values

hours=df['average_montly_hours'].values

years=df['time_spend_company'].values

accident=df['Work_accident'].values

left=df['left'].values

promotion=df['promotion_last_5years'].values

area=df['sales'].values

salary=df['salary'].values



figsize(15, 10)#graphs size



#plt.scatter(level, time,  color='g')



temp = df['satisfaction_level'].value_counts()



temp =pd.to_numeric(temp)



levels=len(temp)



temp = df['last_evaluation'].value_counts()



temp =pd.to_numeric(temp)



times=len(temp)



np.zeros((levels,times), dtype=np.int)





'''

cm = plt.cm.get_cmap('RdYlBu_r')



# Plot histogram.

n, bins, patches = plt.hist(df['satisfaction_level'], 15, color='green')

bin_centers = 0.5 * (bins[:-1] + bins[1:])



# scale values to interval [0,1]

col = bin_centers - min(bin_centers)

col /= max(col)



for c, p in zip(col, patches):

    plt.setp(p, 'facecolor', cm(c))

'''

'''



fig = plt.figure(figsize=(10, 10)) 

fig_dims = (5, 2)



plt.subplot2grid(fig_dims, (0, 0))

#df['satisfaction_level'].hist(bins=15)#sturges' rule



cm = plt.cm.get_cmap('RdYlBu_r')



# Plot histogram.

n, bins, patches = plt.hist(df['satisfaction_level'], 15, color='green')

bin_centers = 0.5 * (bins[:-1] + bins[1:])



# scale values to interval [0,1]

col = bin_centers - min(bin_centers)

col /= max(col)



for c, p in zip(col, patches):

    plt.setp(p, 'facecolor', cm(c))



    

#df['satisfaction_level'].value_counts().plot(kind='bar', title='level')



plt.subplot2grid(fig_dims, (0, 1))

df['last_evaluation'].hist(bins=15)#sturges' rule

#df['last_evaluation'].value_counts().plot(kind='bar', title='time')



plt.subplot2grid(fig_dims, (1, 0))

df['number_project'].hist(bins=6)#different values

#df['number_project'].value_counts().plot(kind='bar', title='projects')



plt.subplot2grid(fig_dims, (1, 1))

df['average_montly_hours'].hist(bins=15)#sturges' rule

#df['average_montly_hours'].value_counts().plot(kind='bar', title='hours')



plt.subplot2grid(fig_dims, (2, 0))

df['time_spend_company'].hist(bins=9)#different values

#df['time_spend_company'].value_counts().plot(kind='bar', title='years')



plt.subplot2grid(fig_dims, (2, 1))

df['Work_accident'].value_counts().plot(kind='bar', title='accident')



plt.subplot2grid(fig_dims, (3, 0))

df['left'].value_counts().plot(kind='bar', title='left')



plt.subplot2grid(fig_dims, (3, 1))

df['promotion_last_5years'].value_counts().plot(kind='bar', title='promotion')



plt.subplot2grid(fig_dims, (4, 0))

df['sales'].value_counts().plot(kind='bar', title='area')



plt.subplot2grid(fig_dims, (4, 1))

df['salary'].value_counts().plot(kind='bar', title='salary')

'''

'''

pd.crosstab(df['left'], df['promotion_last_5years'])



temp = pd.crosstab(df['sales'], df['Work_accident'])



print(temp)



area_accident = []



for i in range(0,9):

    area_accident.append (temp[1][i]/(temp[0][i]+temp[1][i]))

    

df = list(df['sales'].columns.values)



percentages =[ list(df['sales'].columns.values), area_accident]





for i in range(0,14999):

    if (df['salary'].values[i] == 'low'):

        df['salary'].values[i]=1

    elif (df['salary'].values[i] == 'medium'):

        df['salary'].values[i]=2

    else:

        df['salary'].values[i]=3



df['salary']=pd.to_numeric(df['salary'])



df.corr()



pd.crosstab(df['sales'], df['number_project'])

pd.crosstab(df['sales'], df['time_spend_company'])

pd.crosstab(df['sales'], df['Work_accident'])

pd.crosstab(df['sales'], df['left'])

pd.crosstab(df['sales'], df['promotion_last_5years'])

#pd.crosstab(df['sales'], df['salary'])

'''



    
