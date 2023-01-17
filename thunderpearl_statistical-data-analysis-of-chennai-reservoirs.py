# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
reservoir_levels = pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv')


reservoir_rainfalls = pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv')
reservoir_levels.head()
reservoir_rainfalls.head()
# Reading the file with the population info of the 



chennai_population =  pd.read_csv('/kaggle/input/chennai-population/chennai_population.csv')
chennai_population.head()
# importing style

from matplotlib import style



# using the style as ggplot

style.use("ggplot")



plt.figure(figsize=(40,14))



plt.plot(reservoir_levels['POONDI'],'b',linewidth = 3)



plt.xlabel("dates")

plt.ylabel("levels")
plt.figure(figsize=(40,14))



plt.plot(reservoir_rainfalls['POONDI'],'o')



plt.xlabel("Days from year 2004 to 2019")

plt.ylabel("Rainfall in mcft")
plt.figure(figsize=(40,14))

plt.bar(list(range(reservoir_rainfalls.shape[0])),reservoir_rainfalls['POONDI'])

plt.xlabel("Days of rainfall from year 2004 to 2019")

plt.ylabel("Rainfall in mcft")
reservoir_levels['Date'] = pd.to_datetime(reservoir_levels['Date'],format="%d-%m-%Y",dayfirst=True)



reservoir_levels['year'], reservoir_levels['month'] = reservoir_levels['Date'].dt.year, reservoir_levels['Date'].dt.month
reservoir_levels.head()
# Means, data we have till june only for the year of 2019

reservoir_levels.tail()
# Seperating the data on the basis of years

reservoir_levels_2004 = reservoir_levels[reservoir_levels.year==2004]

reservoir_levels_2005 = reservoir_levels[reservoir_levels.year==2005]

reservoir_levels_2006 = reservoir_levels[reservoir_levels.year==2006]

reservoir_levels_2007 = reservoir_levels[reservoir_levels.year==2007]

reservoir_levels_2008 = reservoir_levels[reservoir_levels.year==2008]

reservoir_levels_2009 = reservoir_levels[reservoir_levels.year==2009]

reservoir_levels_2010 = reservoir_levels[reservoir_levels.year==2010]

reservoir_levels_2011 = reservoir_levels[reservoir_levels.year==2011]

reservoir_levels_2012 = reservoir_levels[reservoir_levels.year==2012]

reservoir_levels_2013 = reservoir_levels[reservoir_levels.year==2013]

reservoir_levels_2014 = reservoir_levels[reservoir_levels.year==2014]

reservoir_levels_2015 = reservoir_levels[reservoir_levels.year==2015]

reservoir_levels_2016 = reservoir_levels[reservoir_levels.year==2016]

reservoir_levels_2017 = reservoir_levels[reservoir_levels.year==2017]

reservoir_levels_2018 = reservoir_levels[reservoir_levels.year==2018]

reservoir_levels_2019 = reservoir_levels[reservoir_levels.year==2019]
# 2004 is a leap year

reservoir_levels_2004.shape

# 2005 is not a leap year

reservoir_levels_2005.shape
# plotting for 'POONDI' reservoir for year 2004

plt.figure(figsize=(30,14))

plt.plot(reservoir_levels_2004['POONDI'],'r',linewidth = 4)



plt.xlabel("All 366 days in year 2004")

plt.ylabel("Reservoirs level in mcft")

plt.title("Line graph of resevoir level for POONDI reservoir in year 2004 in MCFT")


reservoir_rainfalls['Date'] = pd.to_datetime(reservoir_rainfalls['Date'],format="%d-%m-%Y",dayfirst=True)





reservoir_rainfalls['year'], reservoir_rainfalls['month'] = reservoir_rainfalls['Date'].dt.year, reservoir_rainfalls['Date'].dt.month
reservoir_rainfalls.head()
reservoir_rainfalls.tail()
# Seperating the reservoir rainfall data on the basis of the year



reservoir_rainfalls_2004 = reservoir_rainfalls[reservoir_rainfalls.year==2004]

reservoir_rainfalls_2005 = reservoir_rainfalls[reservoir_rainfalls.year==2005]

reservoir_rainfalls_2006 = reservoir_rainfalls[reservoir_rainfalls.year==2006]

reservoir_rainfalls_2007 = reservoir_rainfalls[reservoir_rainfalls.year==2007]

reservoir_rainfalls_2008 = reservoir_rainfalls[reservoir_rainfalls.year==2008]

reservoir_rainfalls_2009 = reservoir_rainfalls[reservoir_rainfalls.year==2009]

reservoir_rainfalls_2010 = reservoir_rainfalls[reservoir_rainfalls.year==2010]

reservoir_rainfalls_2011 = reservoir_rainfalls[reservoir_rainfalls.year==2011]

reservoir_rainfalls_2012 = reservoir_rainfalls[reservoir_rainfalls.year==2012]

reservoir_rainfalls_2013 = reservoir_rainfalls[reservoir_rainfalls.year==2013]

reservoir_rainfalls_2014 = reservoir_rainfalls[reservoir_rainfalls.year==2014]

reservoir_rainfalls_2015 = reservoir_rainfalls[reservoir_rainfalls.year==2015]

reservoir_rainfalls_2016 = reservoir_rainfalls[reservoir_rainfalls.year==2016]

reservoir_rainfalls_2017 = reservoir_rainfalls[reservoir_rainfalls.year==2017]

reservoir_rainfalls_2018 = reservoir_rainfalls[reservoir_rainfalls.year==2018]

reservoir_rainfalls_2019 = reservoir_rainfalls[reservoir_rainfalls.year==2019]
plt.figure(figsize=(30,14))

plt.plot(reservoir_rainfalls_2004['POONDI'],'b',linewidth = 4)



plt.xlabel("All 366 days in year 2004")

plt.ylabel("Rainfall in mcft")

plt.title("Line graph of rainfall for POONDI reservoir in year 2004 in MCFT")
plt.figure(figsize=(27,24))



plt.subplot(221)

plt.plot(reservoir_levels_2004['POONDI'],'r', label = "reservoir level",linewidth = 4)

plt.legend()



plt.subplot(222)

plt.plot(reservoir_rainfalls_2004['POONDI'],'o',label = "rainfall",linewidth = 4)

plt.legend()



plt.title("Comparision of 'POONDI' reservoir's levels VS rainfall")


# Setting the figure size

plt.figure(figsize=(27,24))



plt.subplot(431)

plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['POONDI'],'b',label = "year 2004",linewidth = 4)

plt.legend()



plt.subplot(432)

plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['POONDI'],'b',label = "year 2005",linewidth = 4)

plt.legend()



plt.subplot(433)

plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['POONDI'],'b',label = "year 2006",linewidth = 4)

plt.legend()



plt.subplot(434)

plt.plot(list(range(reservoir_levels_2007.shape[0])),reservoir_levels_2007['POONDI'],'b',label = "year 2007",linewidth = 4)

plt.legend()



plt.subplot(435)

plt.plot(list(range(reservoir_levels_2008.shape[0])),reservoir_levels_2008['POONDI'],'b',label = "year 2008",linewidth = 4)

plt.legend()



plt.subplot(436)

plt.plot(list(range(reservoir_levels_2009.shape[0])),reservoir_levels_2009['POONDI'],'b',label = "year 2009",linewidth = 4)

plt.legend()



plt.subplot(437)

plt.plot(list(range(reservoir_levels_2010.shape[0])),reservoir_levels_2010['POONDI'],'b',label = "year 2010",linewidth = 4)

plt.legend()



plt.subplot(438)

plt.plot(list(range(reservoir_levels_2011.shape[0])),reservoir_levels_2011['POONDI'],'b',label = "year 2011",linewidth = 4)

plt.legend()



plt.subplot(439)

plt.plot(list(range(reservoir_levels_2012.shape[0])),reservoir_levels_2012['POONDI'],'b',label = "year 2012",linewidth = 4)

plt.legend()
plt.figure(figsize=(27,24))



plt.subplot(431)

plt.plot(list(range(reservoir_levels_2013.shape[0])),reservoir_levels_2013['POONDI'],'b',label = "year 2013",linewidth = 4)

plt.legend()



plt.subplot(432)

plt.plot(list(range(reservoir_levels_2014.shape[0])),reservoir_levels_2014['POONDI'],'b',label = "year 2014",linewidth = 4)

plt.legend()



plt.subplot(433)

plt.plot(list(range(reservoir_levels_2015.shape[0])),reservoir_levels_2015['POONDI'],'b',label = "year 2015",linewidth = 4)

plt.legend()



plt.subplot(434)

plt.plot(list(range(reservoir_levels_2016.shape[0])),reservoir_levels_2016['POONDI'],'b',label = "year 2016",linewidth = 4)

plt.legend()



plt.subplot(435)

plt.plot(list(range(reservoir_levels_2017.shape[0])),reservoir_levels_2017['POONDI'],'b',label = "year 2017",linewidth = 4)

plt.legend()



plt.subplot(436)

plt.plot(list(range(reservoir_levels_2018.shape[0])),reservoir_levels_2018['POONDI'],'b',label = "year 2018",linewidth = 4)

plt.legend()



plt.subplot(437)

plt.plot(list(range(reservoir_levels_2019.shape[0])),reservoir_levels_2019['POONDI'],'b',label = "year 2019",linewidth = 4)

plt.legend()
plt.figure(figsize=(27,24))



plt.subplot(431)

plt.plot(list(range(reservoir_rainfalls_2004.shape[0])),reservoir_rainfalls_2004['POONDI'],'bo-',label = "year 2004",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(432)

plt.plot(list(range(reservoir_rainfalls_2005.shape[0])),reservoir_rainfalls_2005['POONDI'],'bo-',label = "year 2005",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(433)

plt.plot(list(range(reservoir_rainfalls_2006.shape[0])),reservoir_rainfalls_2006['POONDI'],'bo-',label = "year 2006",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(434)

plt.plot(list(range(reservoir_rainfalls_2007.shape[0])),reservoir_rainfalls_2007['POONDI'],'bo-',label = "year 2007",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(435)

plt.plot(list(range(reservoir_rainfalls_2008.shape[0])),reservoir_rainfalls_2008['POONDI'],'bo-',label = "year 2008",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(436)

plt.plot(list(range(reservoir_rainfalls_2009.shape[0])),reservoir_rainfalls_2009['POONDI'],'bo-',label = "year 2009",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(437)

plt.plot(list(range(reservoir_rainfalls_2010.shape[0])),reservoir_rainfalls_2010['POONDI'],'bo-',label = "year 2010",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(438)

plt.plot(list(range(reservoir_rainfalls_2011.shape[0])),reservoir_rainfalls_2011['POONDI'],'bo-',label = "year 2011",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(439)

plt.plot(list(range(reservoir_rainfalls_2012.shape[0])),reservoir_rainfalls_2012['POONDI'],'bo-',label = "year 2012",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.figure(figsize=(27,24))



plt.subplot(431)

plt.plot(list(range(reservoir_rainfalls_2013.shape[0])),reservoir_rainfalls_2013['POONDI'],'bo-',label = "year 2013",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(432)

plt.plot(list(range(reservoir_rainfalls_2014.shape[0])),reservoir_rainfalls_2014['POONDI'],'bo-',label = "year 2014",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(433)

plt.plot(list(range(reservoir_rainfalls_2015.shape[0])),reservoir_rainfalls_2015['POONDI'],'bo-',label = "year 2015",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(434)

plt.plot(list(range(reservoir_rainfalls_2016.shape[0])),reservoir_rainfalls_2016['POONDI'],'bo-',label = "year 2016",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(435)

plt.plot(list(range(reservoir_rainfalls_2017.shape[0])),reservoir_rainfalls_2017['POONDI'],'bo-',label = "year 2017",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(436)

plt.plot(list(range(reservoir_rainfalls_2018.shape[0])),reservoir_rainfalls_2018['POONDI'],'bo-',label = "year 2018",linewidth = 4)

plt.legend()

plt.ylim((0,310))



plt.subplot(437)

plt.plot(list(range(reservoir_rainfalls_2019.shape[0])),reservoir_rainfalls_2019['POONDI'],'bo-',label = "year 2019",linewidth = 4)

plt.legend()

plt.ylim((0,310))

# width,height

plt.figure(figsize=(27,24))



plt.subplot(321)

plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['POONDI'],'b',label = "POONDI reservoir levels in year 2004",linewidth = 4)

plt.legend()



plt.subplot(322)

plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['CHOLAVARAM'],'r',label = "CHOLAVARAM reservoir levels in year 2004",linewidth = 4)

plt.legend()



plt.subplot(323)

plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['REDHILLS'],'k',label = "REDHILLS reservoir levels in year 2004",linewidth = 4)

plt.legend()



plt.subplot(324)

plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['CHEMBARAMBAKKAM'],'g',label = "CHEMBARAMBAKKAM reservoir levels in year 2004",linewidth = 4)

plt.legend()





plt.title("Comparision of reservoir levels of all the Reservoirs in year 2004")

# width,height

plt.figure(figsize=(27,24))



plt.subplot(321)

plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['POONDI'],'b',label = "POONDI reservoir levels in year 2005",linewidth = 4)

plt.legend()



plt.subplot(322)

plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['CHOLAVARAM'],'r',label = "CHOLAVARAM reservoir levels in year 2005",linewidth = 4)

plt.legend()



plt.subplot(323)

plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['REDHILLS'],'k',label = "REDHILLS reservoir levels in year 2005",linewidth = 4)

plt.legend()



plt.subplot(324)

plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['CHEMBARAMBAKKAM'],'g',label = "CHEMBARAMBAKKAM reservoir levels in year 2005",linewidth = 4)

plt.legend()





plt.title("Comparision of reservoir levels of all the Reservoirs in year 2005")
# width,height

plt.figure(figsize=(27,24))



plt.subplot(321)

plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['POONDI'],'b',label = "POONDI reservoir levels in year 2006",linewidth = 4)

plt.legend()



plt.subplot(322)

plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['CHOLAVARAM'],'r',label = "CHOLAVARAM reservoir levels in year 2006",linewidth = 4)

plt.legend()



plt.subplot(323)

plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['REDHILLS'],'k',label = "REDHILLS reservoir levels in year 2006",linewidth = 4)

plt.legend()



plt.subplot(324)

plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['CHEMBARAMBAKKAM'],'g',label = "CHEMBARAMBAKKAM reservoir levels in year 2006",linewidth = 4)

plt.legend()





plt.title("Comparision of reservoir levels of all the Reservoirs in year 2006")
reservoir_levels_grouped_2004 = reservoir_levels_2004.groupby('month').sum()
reservoir_levels_grouped_2004
plt.figure(figsize=(18,18))



plt.subplot(321)

plt.bar(list(range(1,13)),reservoir_levels_grouped_2004['POONDI'], label = "POONDI-2004")

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()



plt.subplot(322)

plt.bar(list(range(1,13)),reservoir_levels_grouped_2004['REDHILLS'], label = "REDHILLS-2004")

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.subplot(323)

plt.bar(list(range(1,13)),reservoir_levels_grouped_2004['CHEMBARAMBAKKAM'], label = "CHEMBARAMBAKKAM-2004")

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.subplot(324)

plt.bar(list(range(1,13)),reservoir_levels_grouped_2004['CHOLAVARAM'], label = "CHOLAVARAM-2004")

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()
reservoir_levels_grouped_mean_2004 = reservoir_levels_2004.groupby('month').mean()
# Custom Colors for all 12 different months



my_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',

             '#911eb4', '#46f0f0', '#f032e6', '#000000', '#fabebe', '#008080', '#000075']
reservoir_levels_grouped_mean_2004
# It's better to compare the mean, year wise



plt.figure(figsize=(18,18))



plt.subplot(321)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2004['POONDI'], label = "POONDI-2004", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()



plt.subplot(322)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2004['REDHILLS'], label = "REDHILLS-2004", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.subplot(323)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2004['CHEMBARAMBAKKAM'], label = "CHEMBARAMBAKKAM-2004", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.subplot(324)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2004['CHOLAVARAM'], label = "CHOLAVARAM-2004", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.title("Month Wise Mean Distribution of reservoirs levels in 2004")
reservoir_levels_grouped_mean_2005 = reservoir_levels_2005.groupby('month').mean()
reservoir_levels_grouped_mean_2005


plt.figure(figsize=(18,18))



plt.subplot(321)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2005['POONDI'], label = "POONDI-2005", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()



plt.subplot(322)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2005['REDHILLS'], label = "REDHILLS-2005", color = my_colors )

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.subplot(323)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2005['CHEMBARAMBAKKAM'], label = "CHEMBARAMBAKKAM-2005", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.subplot(324)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2005['CHOLAVARAM'], label = "CHOLAVARAM-2005", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.title("Month Wise Mean Distribution of reservoirs levels in 2005")
reservoir_levels_grouped_mean_2006 = reservoir_levels_2006.groupby('month').mean()
reservoir_levels_grouped_mean_2006
# Plotting the reservoir bar for year 2006





plt.figure(figsize=(18,18))



plt.subplot(321)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2006['POONDI'], label = "POONDI-2006", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()



plt.subplot(322)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2006['REDHILLS'], label = "REDHILLS-2006", color = my_colors )

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.subplot(323)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2006['CHEMBARAMBAKKAM'], label = "CHEMBARAMBAKKAM-2006", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.subplot(324)

plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2006['CHOLAVARAM'], label = "CHOLAVARAM-2006", color = my_colors)

plt.xlabel("Months from 1 to 12")

plt.ylabel("Total reservoir level for each Month")

plt.legend()





plt.title("Month Wise Mean Distribution of reservoirs levels in 2006")
chennai_population
chennai_population.dtypes
plt.figure(figsize=(15,15))

plt.plot(chennai_population['Year'],chennai_population['Population'],'-ob')

plt.xlim(1950,2050)

plt.ylim(1491293,15376000)

plt.xlabel("Years")

plt.ylabel("Population")

plt.title("Growth of Population with Respect to Years")

plt.show()
reservoir_levels_data_only = reservoir_levels[['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']]
reservoir_levels_data_only.head()
# Creating the correlation matrix

reservoir_data_corr_matrix = reservoir_levels_data_only.corr()
reservoir_data_corr_matrix
# For better analysis, we will create the heatmap

sns.heatmap(reservoir_data_corr_matrix)

