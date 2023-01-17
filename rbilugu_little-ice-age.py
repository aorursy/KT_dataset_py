%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# coding: utf-8

# ### Goal
# - To do some basic EDA on the data for any interesting weather patterns during the given period. 
# 
# #### Step 1: Setting up the environment
# - Import requred libraries
# - Check/set the options
# - Load data

# In[1]:

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline


# In[2]:

## Check/Set options
## Get show_dimensions/max_rows/max_columns
# pd.get_option('display.show_dimensions')
# pd.get_option('display.max_rows')
# pd.get_option('display.max_columns')
## To get the size of rows and columns
# pd.read_csv('CLIWOC15.csv').shape
pd.set_option('display.show_dimensions', True)
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)


# In[3]:

# Load dataset and check the columns
df = pd.read_csv("../input/CLIWOC15.csv")
df.count()


# ##### Step 2: EDA - As I'm trying to analyse the weather change patterns from the given data, I think I'll need the following data from the dataset
# - Date (Year & Month)
# - Coordinates (Latitude & Longitude)
# - Weather info & measurements
#     - Temperature 
#     - Atmospheric Preasure 
#     - Humidity
#     - Gusts/Rain/Fog/Snow/Thunder/SeaIce/etc
#  
#  For now, will just analyze Air temperature changes over this period

# In[4]:

# working data set df
wdf = df.ix[:,['Year','Month','Lon3','Lat3','TairReading','BaroReading','Rain','Fog','Snow','Thunder','Hail','SeaIce']]
# truncate the rows with NaN Logitude and Latitude values as they are of no use for us for what we are doing
wdf = wdf[wdf.Lon3.notnull() & wdf.Lat3.notnull()]


# In[15]:

fig,ax = plt.subplots(3,2)
fig.set_size_inches(16,12)
xmin = -180; xmax = 180
ymin = -90; ymax = 90
#fig.set_label(['Longitude','Latitude'])
fig.suptitle("Reported Weather Conditions", fontsize=20)

ax[0,0].scatter(wdf[wdf.Rain == 1].Lon3, wdf[wdf.Rain == 1].Lat3, marker='o', s=3, c='b')
ax[0,0].set_title('Rain');ax[0,0].grid();ax[0,0].set_xlim(xmin,xmax);ax[0,0].set_ylim(ymin,ymax);ax[0,0].set_ylabel('Latitude')
ax[0,1].scatter(wdf[wdf.Fog == 1].Lon3, wdf[wdf.Fog == 1].Lat3, marker='o', s=3, c='r')
ax[0,1].set_title('Fog');ax[0,1].grid();ax[0,1].set_xlim(xmin,xmax);ax[0,1].set_ylim(ymin,ymax)
ax[1,0].scatter(wdf[wdf.Snow == 1].Lon3, wdf[wdf.Snow == 1].Lat3, marker='o', s=3, c='b')
ax[1,0].set_title('Snow');ax[1,0].grid();ax[1,0].set_xlim(xmin,xmax);ax[1,0].set_ylim(ymin,ymax);ax[1,0].set_ylabel('Latitude')
ax[1,1].scatter(wdf[wdf.SeaIce == 1].Lon3, wdf[wdf.SeaIce == 1].Lat3, marker='o', s=5, c='b')
ax[1,1].set_title('SeaIce');ax[1,1].grid();ax[1,1].set_xlim(xmin,xmax);ax[1,1].set_ylim(ymin,ymax)
ax[2,1].scatter(wdf[wdf.Thunder == 1].Lon3, wdf[wdf.Thunder == 1].Lat3, marker='o', s=3, c='r')
ax[2,1].set_title('Thunder');ax[2,1].grid();ax[2,1].set_xlim(xmin,xmax);ax[2,1].set_ylim(ymin,ymax);ax[2,1].set_xlabel('Longitude')
ax[2,0].scatter(wdf[wdf.Hail == 1].Lon3, wdf[wdf.Hail == 1].Lat3, marker='o', s=3, c='r')
ax[2,0].set_title('Hail');ax[2,0].grid();ax[2,0].set_xlim(xmin,xmax);ax[2,0].set_ylim(ymin,ymax);
ax[2,0].set_xlabel('Longitude');ax[2,0].set_ylabel('Latitude')


# ##### One interesting point to note from the above plots : Snow and SeaIce close to Equator! We will explore this further, later in the process. 
# 
# ###### Now, lets take a quick look as how the air temperatures look like over the given time period. I grouped this on monthly basis - though initially I thought of grouping them on season (winter/spring/summer/fall) basis, decided to go with monthly as it provides better granularity

# In[6]:

# Now, lets take a quick look at how the monthly mean temps changed over the years
tmp_mean = wdf.groupby(['Month','Year'])['TairReading'].mean()


# In[10]:

#fig,ax = plt.subplots(6,2, sharex = True, sharey = True, )
fig,ax = plt.subplots(6,2)
xmin = 1740; xmax = 1860
fig.suptitle("Mean Air Temperatures", fontsize=20)
fig.set_size_inches(16,24)
ax[0,0].plot(tmp_mean[1]);ax[0,0].set_title('Jan');ax[0,0].grid();ax[0,0].set_xlim(xmin,xmax);ax[0,0].set_ylabel('Temperature (F)')
ax[0,1].plot(tmp_mean[2]);ax[0,1].set_title('Feb');ax[0,1].grid();ax[0,1].set_xlim(xmin,xmax)
ax[1,0].plot(tmp_mean[3]);ax[1,0].set_title('Mar');ax[1,0].grid();ax[1,0].set_xlim(xmin,xmax);ax[1,0].set_ylabel('Temperature (F)')
ax[1,1].plot(tmp_mean[4]);ax[1,1].set_title('Apr');ax[1,1].grid();ax[1,1].set_xlim(xmin,xmax)
ax[2,0].plot(tmp_mean[5]);ax[2,0].set_title('May');ax[2,0].grid();ax[2,0].set_xlim(xmin,xmax);ax[2,0].set_ylabel('Temperature (F)')
ax[2,1].plot(tmp_mean[6]);ax[2,1].set_title('Jun');ax[2,1].grid();ax[2,1].set_xlim(xmin,xmax)
ax[3,0].plot(tmp_mean[7]);ax[3,0].set_title('Jul');ax[3,0].grid();ax[3,0].set_xlim(xmin,xmax);ax[3,0].set_ylabel('Temperature (F)')
ax[3,1].plot(tmp_mean[8]);ax[3,1].set_title('Aug');ax[3,1].grid();ax[3,1].set_xlim(xmin,xmax)
ax[4,0].plot(tmp_mean[9]);ax[4,0].set_title('Sep');ax[4,0].grid();ax[4,0].set_xlim(xmin,xmax);ax[4,0].set_ylabel('Temperature (F)')
ax[4,1].plot(tmp_mean[10]);ax[4,1].set_title('Oct');ax[4,1].grid();ax[4,1].set_xlim(xmin,xmax)
ax[5,0].plot(tmp_mean[11]);ax[5,0].set_title('Nov');ax[5,0].grid();ax[5,0].set_xlim(xmin,xmax);ax[5,0].set_ylabel('Temperature (F)')
ax[5,1].plot(tmp_mean[12]);ax[5,1].set_title('Dec');ax[5,1].grid();ax[5,1].set_xlim(xmin,xmax)
ax[5,0].set_xlabel('Year');ax[5,1].set_xlabel('Year')


# ##### Well, from above plots it appears the temperatures are significantly low throughout the year (all 12 months) between 1780 and 1800.  Not sure if this is due to bad data. 
# Lets see if we can find anything interesting wrt the seaice in the tropics. If we can establish that (seaice in tropiccal seas) phenomenon occured around the same time frame, then we have a solid proof to confirm the data is valid and also some interesting conclusions to make!

# In[14]:

fig = plt.figure()
fig.set_size_inches(16,4)
fig.suptitle("SeaIce/Snow in Tropics", fontsize=20)

xmin = -180; xmax = 180
ymin = -23; ymax = 23
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

ax1.set_xlabel('Latitude');ax1.set_ylabel('Longitude'); ax1.set_title("SeaIce");
ax1.grid();ax1.set_xlim(xmin,xmax);ax1.set_ylim(ymin,ymax)
ax2.set_xlabel('Latitude');ax2.set_ylabel('Longitude'); ax2.set_title("Snow");
ax2.grid();ax2.set_xlim(xmin,xmax);ax2.set_ylim(ymin,ymax)
ax3.set_xlabel('Year');ax3.set_ylabel('Count');ax3.set_title("Sightings per Year");ax3.grid()


fsn = lambda x: ((x.Snow == 1) & (x.Lat3 > ymin) & (x.Lat3 < ymax))
fsi = lambda x: ((x.SeaIce == 1) & (x.Lat3 > ymin) & (x.Lat3 < ymax))

ax1.scatter(wdf[fsi(wdf)].Lon3, wdf[fsi(wdf)].Lat3, marker='o', s=3, c='b')
ax2.scatter(wdf[fsn(wdf)].Lon3, wdf[fsn(wdf)].Lat3, marker='o', s=5, c='b')
ax3.plot(wdf[fsi(wdf)].groupby('Year')['SeaIce'].count())


# ##### Bingo!!! The above plots confirm some 'SeaIce' reported within the tropical region - more predominantly in the Tropic of Cancer, and a significant spike in 'SeaIce' sightings reported in the tropical region between 1790 and 1800. Based on this, a quick googling revealed info about the 'Little Ice Age'
# https://en.wikipedia.org/wiki/Little_Ice_Age
# 
# ##### Step-3 Conclusion - Well, started off on this to test my newly acquired python skills and was pleasantly surprised to discover something I did not know until now! 
# 
# -rbilugu
