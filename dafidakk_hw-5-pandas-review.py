# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe=pd.read_csv('../input/Admission_Predict.csv')
dataframe.head()

dataframe.columns
#building frames from dictionary
team_players=["quaresma","karius","babel","oguzhan","tolgay"]
#column name(feature name)=values   --- created as a list
players_ages=["34","29","33","28","26"]
#column name(feature name)=values   --- created  as a list
list_label=["team_players","players_ages"]
#feature names also created in a list(list_label) as a list
list_column=[team_players,players_ages]
#columns created  as a list
zipped=list(zip(list_label,list_column)) # key point(list(zip))  zip returns tuble we evaluate to list
#zip functions used zipping the lists!!
data_dictionary=dict(zipped)
#for making dataframes first we create dictionary
#then we use pd.DataFrame method making dataframe;
dataframe_player=pd.DataFrame(data_dictionary)
dataframe_player
#adding new columns and assigning different values different entire column

dataframe_player["Nationality"]=["Portugal","Holland","Germany","Turkey","Turkey"]
dataframe_player
#Broadcasting= Creating new column and assigning same value to entire column
dataframe_player["Season_Injuries"]=False
dataframe_player["Injuries"]=False

dataframe_player
#Plotting All Data

dataframe1=dataframe.loc[:,['SOP','LOR ']]
dataframe1.plot()
# it might be confusing

# same data subplot
dataframe1.plot(subplots=True)
plt.show()

#scatter plot
dataframe1.plot(kind="scatter",x='SOP',y='LOR ')
plt.show()
#Histogram plot masures frequance

dataframe.plot(kind='hist',y='CGPA')
#additional features   dataframe.plot(kind='hist',y='CGPA',bins=25,range=(0,250),normed=True)
                                                        # bins= figure thikness,range= x label scale,normed=(ylabel) frequance normalized(True or False)

fig, axes =plt.subplots(nrows=2,ncols=1)  # this code line represant, output plots line in a rows and columns
dataframe.plot(kind='hist',y='CGPA',bins=25,range=(0,15),normed=True,ax=axes[0]) # in place 0th row
dataframe.plot(kind='hist',y='CGPA',bins=25,range=(0,15),normed=True,ax=axes[1],cumulative=True) # in place 1th row
plt.savefig('graph.png')
plt.show()
date_time_list=["2017-01-01","2017-12-31"] # this is string type of date time
print(type(date_time_list[1])) #date_time_list's [1] value is str,
#to convert datetime object;
datetime_object=pd.to_datetime(date_time_list)
print(type(datetime_object)) # type = pandas date time index
dataframe_player=dataframe_player.drop("Injuries",axis=1)
dataframe_player
date_list_season=["2018-01-01","2018-02-01","2018-04-01","2018-06-01","2018-12-01"] # craating time series list
date_time_season_object=pd.to_datetime(date_list_season) # converting pandas datetime object
dataframe_player["Season"]=date_time_season_object  # labeling with Season
dataframe_player
dataframe_player=dataframe_player.set_index("Season")  # set as a index of the dataframe_player
dataframe_player
print(dataframe_player.loc["2018-01-01"])  # slicing
print(dataframe_player.loc["2018-01-01":"2018-06-01"]) # slicing
dataframe_player["players_ages"]=dataframe_player["players_ages"].astype("int")
#For example
#dataframe_player.resample("Y").mean()  # for each Year mean speratly calculated. same "A" is the same key word for year
dataframe_player.resample("M").mean()   # for each Mounth's mean speratly calculated.
#dataframe_player["players_ages"].resample("M").mean() #  gives just players_ages resample
# in output data, there are some NaN value , even if in the main data hasn't that spesicif Mounth.
# 'M' key word gives first mount.

# last tip: out last time index is 2018-12-01 but M resample output gives 2018-12-31, 31 is the last day of the 12th mounth.
#dataframe_player.resample("D").mean() # this line give days but the days which has no day is going to be NaN and the whole year!! :)
dataframe_player.resample("M").first().interpolate("linear")
 #interpolate by the first value to last. fillling with between two values rates and directions. for example between karius and babel age is increasing interpolate,but between babel and oguzhan
    # decreasing so on..
dataframe_player.resample("M").mean().interpolate("linear") # interpolating with mean
