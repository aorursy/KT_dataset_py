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
import pandas as pd

life_vs_work = pd.read_csv("../input/life-expectancy-and-hours-worked-by-country/Life expectancy vs Working Hours.csv")
import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt
#getting a sense of what the data in my CSV look like.

life_vs_work.head()

# PLOT 1

#An overview of what this data set looks like. Most of the average hours are between 35-40 per week,

#but some countries have averages as high as 55 or 56.



hist=life_vs_work.hist(column="Working Hours")
# PLOT 2

#An overview of what this data set looks like. Most of the life expectancies are between 75 and 81 years old,

#but some countries have life expectancies as low as 65.



hist=life_vs_work.hist(column="Life Expectancy")
# PLOT 3

#Trying an initial scatterplot looking at working hours by year for the different countries. In general, 

#the trend seems to be going down or staying relatively the same.



x = life_vs_work[['Year']]

y = life_vs_work[['Working Hours']]



plt.style.use("ggplot")



life_vs_work.plot(kind = 'scatter', x ='Year', y = "Working Hours")



plt.title("Working Hours by Year for Nine Countries")

plt.xlabel("Year")

plt.ylabel("Average Working Hours")

# PLOT 4

#Attempting to plot working hours over the years in different colors, with no luck

#Trying to do this with a for loop iterating over a list, and using groupby and legends



#countries = ["Canada", "Chile", "Costa Rica","Israel", "Philippines","South Korea", "Spain", "Sweden", "United States"]



#for country in countries:

   # x = life_vs_work[['Year']]

   # y = life_vs_work[['Working Hours']]

   # plt.plot(life_vs_work[life_vs_work["Country"]] == country,kind = 'line', x ='Year', y = "Working Hours")





#for idx, gp in life_vs_work.groupby('Country'):

    #plt.plot(gp, x='Year', y='Working Hours', label=idx)





#plt.legend()





    
# PLOT 5

#Trying an initial scatterplot looking at life expectancy by year for the different countries. This time,  

#the trend seems to be going up for every country



x = life_vs_work[['Year']]

y = life_vs_work[['Life Expectancy']]



plt.style.use("ggplot")



life_vs_work.plot(kind = 'scatter', x ='Year', y = "Life Expectancy")



plt.title("Life Expectancy Over Time Nine Countries")

plt.xlabel("Year")

plt.ylabel("Average Life Expectancy")
# PLOT 6

#Now examining how working hours and life expectancy might correlate.  As the trend on the

#scatterplot clearly shows, as working hours increase, life expectancy tends to decrease.

#The highest life expectancies seem to be grouped around 36-38 hours of work/week in this data set.

#The lowest are grouped around 47 hours of work/week.



x = life_vs_work[['Working Hours']]

y = life_vs_work[['Life Expectancy']]





life_vs_work.plot(kind = 'scatter', x ='Working Hours', y = "Life Expectancy")



plt.title("Life Expectancy vs Hours Worked Over Time")

plt.xlabel("Average Working Hours")

plt.ylabel("Average Life Expectancy")