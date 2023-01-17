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
movies = pd.read_csv("../input/IMDB-Movie-Data.csv")

movies.head()
# information about number of rows and columns

movies.shape
#check the min ,max, count of movies

movies.describe()
# checking for null values

movies.isnull().any()
# dropping null value columns to avoid errors

movies.dropna(inplace = True)



#checking again for null values and got False for all columns

movies.isnull().any()
high_revenue = movies[movies['Revenue (Millions)'] > 550.00]

high_revenue.head()



high_revenue.insert(11,'Box office (Billions)', [2.06, 1.51, 1.67, 2.78])

high_revenue_sorted = high_revenue.sort_values(by = 'Year')

high_revenue_sorted
import matplotlib.pyplot as plt
# Plotting year vs revenue

year = high_revenue_sorted['Year'].values

revenue = high_revenue_sorted['Revenue (Millions)'].values



plt.plot(year, revenue)

plt.xlabel("Year")

plt.ylabel("Revenue (Millions)")

plt.show()
# Plotting the highest revenue and annotating the high revenue movie

fig,ax = plt.subplots()

ax.annotate(high_revenue_sorted['Title'].iloc[0],

            xy = (2009,760), xycoords = 'data',

            xytext = (2009,780), textcoords = 'data',

            arrowprops = dict(arrowstyle = '->',

                             connectionstyle = 'arc3'),

           )



ax.annotate(high_revenue_sorted['Title'].iloc[1],

            xy = (2012,623), xycoords = 'data',

            xytext = (2012,650), textcoords = 'data',

            arrowprops = dict(arrowstyle = '->',

                             connectionstyle = 'arc3'),

           )



ax.annotate(high_revenue_sorted['Title'].iloc[2],

            xy = (2015,936), xycoords = 'data',

            xytext = (2015,950), textcoords = 'data',

            arrowprops = dict(arrowstyle = '->',

                             connectionstyle = 'arc3'),

           )



ax.annotate(high_revenue_sorted['Title'].iloc[3],

            xy = (2015,652), xycoords = 'data',

            xytext = (2015,630), textcoords = 'data',

            arrowprops = dict(arrowstyle = '->',

                             connectionstyle = 'arc3'),

           )





year = high_revenue_sorted['Year'].values

revenue = high_revenue_sorted['Revenue (Millions)'].values



plt.plot(year, revenue)

plt.xlabel("Year")

plt.ylabel("Revenue (Millions)")

#plt.axis([2009,2015,0,950])

plt.show()