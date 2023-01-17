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
import matplotlib.pyplot as plt

import seaborn as sns
data19 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')

data20 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')
data19.drop('Unnamed: 21', axis = 1, inplace = True)

data20.drop('Unnamed: 21', axis = 1, inplace = True)
fig, ax = plt.subplots(figsize = (10,5))

X = range(len(data19['CANCELLED'].value_counts()))

new_val = []

for item in X:

    new_val.append(item+0.25)

plt.title('Bar chart showing comparisons between Janurary 2019 and Janurary 2020')

bar1 = plt.bar(X, data19['CANCELLED'].value_counts(), width = 0.25)

bar2 = plt.bar(new_val, data20['CANCELLED'].value_counts(), width= 0.25)

plt.xticks([0.12, 1.12], ['Departed', 'Cancelled'])

plt.yticks(data20['CANCELLED'].value_counts(), " ")

ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)

ax.spines["left"].set_visible(False)



ax.tick_params(axis = "both", labelsize = 15)



def autolabel(rects, xpos  = 'center'):

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}

    offset = {'center': 0, 'right': 1, 'left': -1}

    

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(offset[xpos]*3, 3),  # use 3 points offset

                    textcoords="offset points",  # in both directions

                    ha=ha[xpos], va='bottom')



autolabel(bar1)

autolabel(bar2)

plt.legend(["Jan19", "Jan20"],frameon = False )

plt.show()
plt.subplot(121)

plt.title('Pie chart showing Percentage of Cancelled Flights for Jan19')

plt.pie(data19['CANCELLED'].value_counts(), labels = ['Departed', 'Cancelled'], radius = 1.25, autopct = '%1.2f%%')

plt.subplots_adjust(left  = 0.125, right = 1.8, bottom = 0.1, top = 0.9, wspace = 0.7, hspace = 0.7)

plt.subplot(122)

plt.title('Pie chart showing Percentage of Cancelled Flights for Jan20')

plt.pie(data20['CANCELLED'].value_counts(), labels = ['Departed', 'Cancelled'], radius = 1.25, autopct = '%1.2f%%')

plt.show()
cancel_df19 = data19[data19['CANCELLED'] == 1.0]

cancel_df20 = data20[data20['CANCELLED'] == 1.0]

group19 = cancel_df19['CANCELLED'].groupby(cancel_df19['ORIGIN']).count()

group20 = cancel_df20['CANCELLED'].groupby(cancel_df20['ORIGIN']).count()



fig, ax = plt.subplots(figsize = (10,5))

bar1 = plt.bar(group19.sort_values(ascending = False).index[:10], group19.sort_values(ascending = False)[:10])

# plt.bar(group20.sort_values(ascending = False).index[:10], group20.sort_values(ascending = False)[:10])

bar2 = plt.bar(group19.sort_values(ascending = False).index[:10], group20.sort_values(ascending = False)[:10])

plt.title('Top 10 Origin Airport where Maximum Flights got cancelled\n Based on JAN19 Data')

ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)

ax.spines["left"].set_visible(False)

plt.yticks([0], " ")

ax.tick_params(axis = "both", labelsize = 15)





def autolabel(rects, xpos  = 'center'):

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}

    offset = {'center': 0, 'right': 1, 'left': -1}

    

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(offset[xpos]*3, 3),  # use 3 points offset

                    textcoords="offset points",  # in both directions

                    ha=ha[xpos], va='bottom')

plt.legend(['JAN19', 'JAN20'])



autolabel(bar1)

autolabel(bar2)
fig, ax = plt.subplots(figsize = (10,5))



group19 = cancel_df19['CANCELLED'].groupby(cancel_df19['DEST']).count()

group20 = cancel_df20['CANCELLED'].groupby(cancel_df20['DEST']).count()



bar1 = plt.bar(group19.sort_values(ascending = False).index[:10], group19.sort_values(ascending = False)[:10])

bar2 = plt.bar(group19.sort_values(ascending = False).index[:10], group20.sort_values(ascending = False)[:10])

plt.title('Top 10 Destination Airport where Maximum Flights got cancelled\n Based on JAN19 Data')



ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)

ax.spines["left"].set_visible(False)

plt.yticks([0], " ")

ax.tick_params(axis = "both", labelsize = 15)



plt.legend(['JAN19', 'JAN20'])



def autolabel(rects, xpos  = 'center'):

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}

    offset = {'center': 0, 'right': 1, 'left': -1}

    

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(offset[xpos]*3, 3),  # use 3 points offset

                    textcoords="offset points",  # in both directions

                    ha=ha[xpos], va='bottom')



autolabel(bar1)

autolabel(bar2)