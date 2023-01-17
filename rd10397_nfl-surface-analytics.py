#imports 

import pandas as pd

import re

import numpy as np

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



        



#InjuryRecord = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

#PlayList = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

#PlayerTrackData = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")        



# Any results you write to the current directory are saved as output.
#reading files

#1

playerTrackData=pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")



#2

playList=pd.read_csv(r"../input/nfl-playing-surface-analytics/PlayList.csv")

#3

injuryRecord=pd.read_csv(r"../input/nfl-playing-surface-analytics/InjuryRecord.csv")
#File information

print(playerTrackData.info())

print(playList.info())

print(injuryRecord.info())
##################################playerTrackData###########################################

play=playerTrackData.iloc[:,0]

play_gameid=[]

for i in play:

    gameid_regex=re.findall("^[0-9]*-[0-9]*",i)

    play_gameid.extend(gameid_regex)

playerTrackData['GameId']=play_gameid

playerTrackData['event']=playerTrackData['event'].fillna(method='ffill')

playerTrackData['event']=playerTrackData['event'].replace(to_replace=['play_submit\t'],value="play_submit")

del play

del play_gameid
##################################playList###########################################



playList['StadiumType']=playList['StadiumType'].replace(to_replace=["Outdoors","Oudoor","Ourdoor","Outddors"\

                                                                    ,"Outdor","Outside","Heinz Field","Bowl","Cloudy"],value="Outdoor")

playList['StadiumType']=playList['StadiumType'].replace(to_replace=["Indoors","Indoor","Indoor, Roof Closed"],value="Indoor Closed") 

playList['StadiumType']=playList['StadiumType'].replace(to_replace=["Domed, closed","Dome, closed","Domed, open","Domed, Open","Dome"],value="Domed")

playList['StadiumType']=playList['StadiumType'].replace(to_replace=["Retr. Roof-Closed","Retr. Roof - Closed","Retr. Roof Closed"],value="Retractable Closed")

playList['StadiumType']=playList['StadiumType'].replace(to_replace=["Outdoor Retr Roof-Open","Retr. Roof - Open","Retr. Roof-Open","Retractable Roof"],value="Retractable Open")

playList['StadiumType']=playList['StadiumType'].replace(to_replace=["Indoor, Open Roof","Open"],value="Indoor Open")

playList['Weather']=playList['Weather'].replace(to_replace=["Sunny",'Mostly sunny','Clear to Partly Cloudy','Sun & clouds','Clear and sunny','Clear and Sunny','Sunny, Windy','Mostly Sunny Skies','Sunny, highs to upper 80s','Heat Index 95','Partly clear','Fair','Sunny Skies','Clear skies','Sunny and clear','Partly sunny',"Clear Skies","Mostly Sunny","Partly Sunny",'Clear and warm',"Sunny and warm","Controlled Climate"],value="Clear")

playList['Weather']=playList['Weather'].replace(to_replace=['Clear and Cool','Sunny and cold','Cold','Clear and cold'],value="Clear and Cold")

playList['Weather']=playList['Weather'].replace(to_replace=["Partly Cloudy",'Mostly Coudy','Overcast','cloudy','Coudy','Partly Clouidy','Hazy','Party Cloudy','Partly cloudy','Cloudy and cold','Cloudy and Cool',"Mostly cloudy",'Mostly Cloudy',"Cloudy, fog started developing in 2nd quarter","Cloudy"],value="Cloudy")

playList['Weather']=playList['Weather'].replace(to_replace=['Showers','Rain shower','Cloudy, Rain','Light Rain','Rainy','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.','Scattered Showers'],value="Rain")

playList['Weather']=playList['Weather'].replace(to_replace=['Rain Chance 40%','Cloudy, chance of rain','30% Chance of Rain','10% Chance of Rain','Cloudy, 50% change of rain','Rain likely, temps in low 40s.'],value="Rain Chance")

playList['Weather']=playList['Weather'].replace(to_replace=['Heavy lake effect snow','Cloudy, light snow accumulating 1-3\"'],value="Snow")

playList['Weather']=playList['Weather'].replace(to_replace=['Indoor',"Indoors",'N/A Indoor',"N/A (Indoors)"],value="Unknown")

playList['PlayType'].replace('0',np.nan,inplace=True)

playList['Position'].replace('Missing Data',np.nan,inplace=True)





##################################injuryRecord###########################################



injuryRecord['DM_M28']=injuryRecord["DM_M28"]-injuryRecord["DM_M42"]

injuryRecord['DM_M7']=injuryRecord["DM_M7"]-injuryRecord["DM_M42"]

injuryRecord['DM_M1']=injuryRecord["DM_M1"]-injuryRecord["DM_M42"]

injuryRecord['DM_M7']=injuryRecord["DM_M7"]-injuryRecord["DM_M28"]

injuryRecord['DM_M1']=injuryRecord["DM_M1"]-injuryRecord["DM_M28"]

injuryRecord['DM_M1']=injuryRecord["DM_M1"]-injuryRecord["DM_M7"]