# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #pretty graphs

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#read csv and take a look

df = pd.read_csv('../input/arrests.csv')

df.head(5)

#each sector

df1 = df.drop('State/Territory',axis=1)

df1 = df1.drop([4,13,23,24])

#totals

total = df.loc[[4,13,23,24]]

#start small to test graphs

test = df.loc[:4]

test = test[[0,1,3,4]]

test.index = [1,2,3,4,5]

test
N=5

ind = np.arange(N) # the x locations for the groups

width = 0.25       # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(ind, test.ix[:,3], width, color='r')

rects2 = ax.bar(ind + width, test.ix[:,2], width, color='y')



# add some text for labels, title and axes ticks

ax.set_ylabel('# of Immigrants')

ax.set_title('Illegal Immigrants on our Coastal Border')

ax.set_xticks(ind + width)

ax.set_xticklabels(test['Sector'], rotation='vertical')

ax.legend((rects1[0], rects2[0]), ('Mexicans Only', 'All Illegal Immigrants'), loc='upper left')



def autolabel(rects):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., height+750,

                '%d' % int(height),

                ha='center', va='bottom', rotation = 'vertical')



autolabel(rects1)

autolabel(rects2)

plt.show()
north = df.loc[df['Border'] == 'North']

coast = df.loc[df['Border'] == 'Coast']

sw = df.loc[df['Border'] == 'Southwest']

year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]



northall = north.filter(like='All Illegal Immigrants')

coastall = coast.filter(like='All Illegal Immigrants')

swall = sw.filter(like='All Illegal Immigrants')



northmex = north.filter(like='Mexican')

coastmex = coast.filter(like='Mexican')

swmex = sw.filter(like='Mexican')



coast
N=17

ind = np.arange(N) # the x locations for the groups

width = 0.35       # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(ind, coastmex.ix[4,0:17], width, color='r')

#rects2 = ax.bar(ind + width, coastall.ix[4,0:17], width, color='y')



# add some text for labels, title and axes ticks

ax.set_ylabel('# of Immigrants')

ax.set_title('Illegal Immigrants on our Coastal Border')

ax.set_xticks(ind + width)

ax.set_xticklabels(year, rotation = 'vertical')

ax.legend((rects1[0], rects2[0]), ('Mexicans Only', 'All Illegal Immigrants'), loc='upper right')



def autolabel(rects):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., height+750,

                '%d' % int(height),

                ha='center', va='bottom', rotation = 'vertical')



autolabel(rects1)

#autolabel(rects2)

plt.show()