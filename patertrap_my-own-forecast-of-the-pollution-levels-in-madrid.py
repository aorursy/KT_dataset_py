#%matplotlib  inline

import numpy as np 
import pandas as pd
import glob
import folium #For maps

import matplotlib.pyplot as plt
import seaborn as sns #This is for cool plots. Recommendation from Linda

import missingno as missing #This is for missing data
from fbprophet import Prophet

from datetime import datetime, timedelta

#This is my excuse to learn the basics of time series
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import statsmodels.api as sm
from itertools import product
from math import sqrt
from sklearn.metrics import mean_squared_error

print("Everything imported correctly")
#I'm not used to this system. In kaggle use as path: "../input/filename.csv"
#AVOID the name of the folder.

#The stations file I know it because it's the only one I can open with Excel. The other ones are too big
stations=pd.read_csv("../input/stations.csv")
#stations.head()

#To read the ones from the years, use a loop with glob. Great invention.

path_years="../input/csvs_per_year/csvs_per_year/" #For simplicity later
all_files=glob.glob(path_years+"/*.csv") #DON'T FORGET THE /!!!!!

#Now I allocate the data frames and lists and everything I need
frame=pd.DataFrame()
ls=[] #List is a reserved word in python, I can't name my list as "list" so I put ls

#I read the files now. For every file in the files directory with a similar name, read them and append them to my data frame

#For memory, header=0 allows me to change the columns names later on.
#index_col=False will evaluate as 0. Don't do it, it's ambiguous.
for file_1 in all_files:
    df=pd.read_csv(file_1,index_col=None, header=0)
    ls.append(df)
#Concat places the second df below the first one and so on. It rewrites the index as I would normally do    
frame=pd.concat(ls)
#Have a look to see what I got
frame.head()
#Comment the head to avoid many lines in the output
#Looks ok so far, with a lot of NaNs. Let's worry about it in a couple of cells

#I could print all, but just 5 will do. I have the addresses, which I understand, but the maps use lat/lon
stations.head()

#Select the lat and lon columns to have the positions and pass them to folium.

#Inner bracket is for the list, outer bracket is for indexing
locations=stations[['lat','lon']]
location_list=locations.values.tolist() #Take the locations, with the values, make a list
#Check the difference when printing locations and location_list locations only prints ['lat','lon']
#location_list prints the values

popup=stations['name']

#Again, map is a reserved word
#Make an empty map, with the zoom and the centering
map_stations=folium.Map(location=[40.44, -3.7], zoom_start=11)

#For every point, add a marker. The points start at 0 indexing and finish at the end of the list
# so, put a marker at every location[point], which is already containing stations['lat'],stations['lon']
for point in range(0,len(location_list)):
    folium.Marker(location_list[point], popup=popup.iloc[point]).add_to(map_stations)
#Now show the maps
map_stations
#El carte inglés?? hahhaha

#Don't forget the ; !!! Otherwise it only gives some memory address or something.
missing.matrix(frame);
#This is slow, so try not to run it too much. White blocks are missing data (NaN) and black is some data.
#Black lines/blocks may also be incorrect. Careful with that. The number on the bottom left is the amount 
#This visualizes the amount of non-nulls. Black is data, you can see it in the date and location column
missing.bar(frame);
#In my case this is useless but it's still very cool. It shows how the nullity of one variable is correlated with other variables
# That is, shows if the absence of a variable is usually correlated with the absence or presence of another one
#Useful? Yes, probably. Cool? Yes!
#Commented for speed
#missing.heatmap(frame);
#frame.head() to check the name of the columns. I probably forget again along the way

#Take only the relevant part of the frame dataframe
cols=['date','station','O_3']
o3=frame[cols]

#Convert to ppb and put date and time in something that python understands and ignore the warnings.
o3['date']=pd.to_datetime(o3['date'])
o3.loc[:,'ppb']=24.45*o3.loc[:,'O_3']/48
o3.head()

#Let's see how many stations were active along time. I remember they remove some when the EU was complaining.


plt.rcParams["figure.figsize"] = [16,9] #For good looking plots in modern screens
plt.plot(o3.groupby(['date']).station.nunique());
plt.ylabel('Number of stations operating');
plt.xlabel('Year');
#Sort the df by number of NaNs, print and then I'll select what I want.
#First, see the shape of the df, to see how many rows we have in total. It's about 3.8e5

o3.shape #Don't take the frame variable, o3 is the same but much smaller

#Now group them by most real values and then sort them

#Careful with this! from left to right, grouped_df is the df made by taking the o3 df. Then you group by station. And then, for every station, you count all the O_3 data
#NaN are ignored.

grouped_stations=o3.groupby('station').O_3.count()
#grouped_stations.head()

#If I take grouped_stations and sort, there is one column with the counts, not the O_3 name. Therefore, sort_values ONLY TAKES THE ARGUMENTS ascending, NOT THE COLUMN (AXIS)
sorted_stations=grouped_stations.sort_values(ascending=False)
sorted_stations.head()


#And then figure out which one is the best station. The column is not named station, but id
#I will print the row that matches id to my best station.
stations[stations.id == sorted_stations.index[0]]
      
      

o3_PN=o3[o3.station==sorted_stations.index[0]]
#o3_PN.head() #To check that it is ok

#But I need to work in ppb, remember that. And I said I would use 8 h moving averages.

#Create a new column, which is the ppb, taking the 8 values window and calculate the mean at each time.
#Obviously, sort them by date/hour, otherwise the rolling average is meaningless
o3_PN=o3_PN.sort_values(['date'])
o3_PN.loc[:,'ppb_moving']=o3_PN.loc[:,'ppb'].rolling(8).mean()
#o3_PN.head(15) #The first values are NaN obviously.

o3_PN=o3_PN.sort_values(['date'])
#o3_PN.head(15)
#I can't fill the NaNs with 0, otherwise this would introduce fake data. I'll fill them with the average of the column

o3_PN['ppb_moving']=o3_PN['ppb_moving'].fillna(o3_PN['ppb_moving'].mean()/2)
o3_PN.head(15)

y=o3_PN['ppb_moving']
y=y.reset_index(drop=True) #It drops one column, saving memory
#y=y.drop(columns=['index'], axis=1)

x=np.linspace(0,len(y),num=len(y))
#len(x)
#len(y)


#Plot with sns. First time

#ax=sns.scatterplot(x=y.index,y=y)
#ax.axes.set_xlim(y.index.min(),y.index.max());

#I still like matplotlib better. Probably because it's almost matlab

#plt.scatter(o3_PN.index,y);

#Since there are too many points, I'll only plot a few. Maybe between 10e3 and 15e3

plt.scatter(x,y);
plt.xlim(0, 6e3);

