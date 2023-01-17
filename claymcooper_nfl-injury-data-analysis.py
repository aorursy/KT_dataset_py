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

InjuryRecord = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

PlayList = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

PlayerTrackData = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns



injury = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

playlist = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")



# Phase I: Data Merge - Most of the data needed is on the above named datasets. However, there are some holes, specifically

# in the PlayKey column. To fix it, I merged the files on the PlayKey column and then again on the GameID column to add the

# data to the set where PlayKey was null



result = pd.merge(injury, 

                 playlist,

                 on= 'PlayKey',

                 how ='left')



results =result.drop(['PlayerKey_y', 'GameID_y'],axis=1)

results.rename(columns={'PlayerKey_x':'PlayerKey', 'GameID_x':'GameID'}, inplace=True)



# After the first merge, all the other data can be based around the GameID column since it contains both the Player and Game IDs



playlist2 = playlist.drop_duplicates(subset ="GameID", 

                     keep = 'first')



result2 = pd.merge(results, 

                 playlist2,

                 on= 'GameID',

                 how ='left')



# This final piece eliminates the null data from the first merge and cleans up the column names



results_final =result2.drop(['PlayerKey_y', 'PlayKey_y', 'PlayerDay_x',

                             'PlayerGame_x', 'StadiumType_x', 'FieldType_x',

                             'Temperature_x','Weather_x', 'PlayType_y',

                             'PlayerGamePlay_y', 'Position_x', 'PositionGroup_x', 'RosterPosition_x'],axis=1)



results_final.rename(columns={'PlayerKey_x': 'PlayerKey', 'PlayKey_x': 'PlayKey', 'PlayerDay_y': 'PlayerDay',

                             'PlayerGame_y': 'PlayerGame', 'StadiumType_y': 'StadiumType', 'FieldType_y': 'FieldType',

                             'Temperature_y': 'Temperature','Weather_y': 'Weather', 'PlayType_x': 'PlayType',

                             'PlayerGamePlay_x':'PlayerGamePlay', 'Position_y': 'Position',

                             'PositionGroup_y':'PositionGroup', 'RosterPosition_y':'RosterPosition'}, inplace=True)

print(results_final.head())

# Phase II: Data Clean-up - Two columns (StadiumType and Weather) do not have standard reporting system. Data in these two

# columns will be reclassified to yield better interpretation of the data.



# Weather Category: Cloudy

results_final['Weather'].replace('Partly Cloudy', 'Cloudy', inplace=True)

results_final['Weather'].replace('Coudy', 'Cloudy', inplace=True)

results_final['Weather'].replace('Party Cloudy', 'Cloudy', inplace=True)

results_final['Weather'].replace('Cloudy and Cool', 'Cloudy', inplace=True)

results_final['Weather'].replace('Cloudy, 50% change of rain', 'Cloudy', inplace=True)

results_final['Weather'].replace('Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.', 'Cloudy', inplace=True)

results_final['Weather'].replace('Mostly cloudy', 'Cloudy', inplace=True)

results_final['Weather'].replace('Sun & clouds', 'Cloudy', inplace=True)



# Weather Category: Clear

results_final['Weather'].replace('Clear and warm', 'Clear', inplace=True)

results_final['Weather'].replace('Sunny', 'Clear', inplace=True)

results_final['Weather'].replace('Mostly sunny', 'Clear', inplace=True)

results_final['Weather'].replace('Mostly Sunny', 'Clear', inplace=True)

results_final['Weather'].replace('Sunny and clear', 'Clear', inplace=True)

results_final['Weather'].replace('Clear and Sunny', 'Clear', inplace=True)

results_final['Weather'].replace('Clear skies', 'Clear', inplace=True)

results_final['Weather'].replace('Clear Skies', 'Clear', inplace=True)

results_final['Weather'].replace('Clear and cold', 'Clear', inplace=True)

results_final['Weather'].replace('Cold', 'Clear', inplace=True)

results_final['Weather'].replace('Fair', 'Clear', inplace=True)



# Weather Category: Rain

results_final['Weather'].replace('Rain shower', 'Rain', inplace=True)

results_final['Weather'].replace('Light Rain', 'Rain', inplace=True)



# Weather Category: Indoors

results_final['Weather'].replace('Indoor', 'Indoors', inplace=True)

results_final['Weather'].replace('N/A (Indoors)', 'Indoors', inplace=True)

results_final['Weather'].replace('Controlled Climate', 'Indoors', inplace=True)



# Weather Category: Unknown

results_final['Weather'].fillna('Unknown', inplace=True)



# Stadium Category: Indoors

results_final['StadiumType'].replace('Indoor', 'Indoors', inplace=True)

results_final['StadiumType'].replace('Dome', 'Indoors', inplace=True)

results_final['StadiumType'].replace('Retr. Roof-Closed', 'Indoors', inplace=True)

results_final['StadiumType'].replace('Closed Dome', 'Indoors', inplace=True)

results_final['StadiumType'].replace('Domed, closed', 'Indoors', inplace=True)

results_final['StadiumType'].replace('Retr. Roof - Closed', 'Indoors', inplace=True)

results_final['StadiumType'].replace('Retractable Roof', 'Indoors', inplace=True)

results_final['StadiumType'].replace('Indoor, Roof Closed', 'Indoors', inplace=True)



# Stadium Category: Outdoors

results_final['StadiumType'].replace('Outdoor', 'Outdoors', inplace=True)

results_final['StadiumType'].replace('Retr. Roof - Open', 'Outdoors', inplace=True)

results_final['StadiumType'].replace('Indoor, Open Roof', 'Outdoors', inplace=True)

results_final['StadiumType'].replace('Oudoor', 'Outdoors', inplace=True)

results_final['StadiumType'].replace('Outddors', 'Outdoors', inplace=True)

results_final['StadiumType'].replace('Open', 'Outdoors', inplace=True)

results_final['StadiumType'].replace('Heinz Field', 'Outdoors', inplace=True)



# Stadium Category: Unknown

results_final['StadiumType'].fillna('Unknown', inplace=True)



# Other Clean-up

results_final['Temperature'].replace(-999, 'Unknown', inplace=True)

results_final.fillna('Unknown', inplace=True)



print(results_final)
# Phase III Data Analysis

# Question 1: Do the characteristics of the stadium influence injuries?



color = ["#013369", "#D50A0A", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]



plt.figure(figsize = (15,5), facecolor = "white",)

plt.rcParams['font.size'] = 15

plt.subplot(1, 2, 1,).title.set_text("Field Surface")

sns.set_style("whitegrid")

results_final['Surface'].value_counts().plot.pie(autopct='%.1f%%', colors = ['#013369', '#D50A0A'],textprops=dict(color="w", weight ="bold"), shadow=True, explode =(0.05, 0.05),)

plt.legend(bbox_to_anchor=(0.87,0.2))

plt.ylabel("Percentage of Injuries")



plt.subplot(1, 2, 2,).title.set_text("Stadium Type")

sns.set_style("whitegrid")

sns.countplot(x = 'StadiumType', hue='Surface', data=results_final, palette = sns.color_palette(color),)

plt.ylabel("Number of Injuries")

plt.xlabel(" ")



plt.text(x = -3.3, y = 60.2, s = "Do Stadium Characteristics Influence Injuries?",

               fontsize = 26, color = "black", weight = 'bold', alpha = .75)

plt.text(x = -3.3, y = 56.9, 

               s = "Comparison between Field Surfaces and Stadium Types on Overall Injuries ",

              fontsize = 15, color = "black", alpha = .85)

plt.text(x = -3.8, y = -10, s = 'Source: NFL 1st and Future - Playing Surface Analytics                            https://www.kaggle.com/c/NFL-playing-surface-analytics',fontsize = 14, color = 'white', backgroundcolor = 'gray')

color = ["#013369", "#D50A0A", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

color2 = ["#241075", "#BC9428", "#A5ACAF", "#66C010", "#4790DE", "#2ecc71"]



plt.figure(figsize = (15,5), facecolor = "white",)

plt.rcParams['font.size'] = 15

plt.subplot(1, 2, 1,).title.set_text("Player Position")

sns.set_style("whitegrid")

sns.countplot(x = 'RosterPosition', hue='Surface', data=results_final, palette = sns.color_palette(color),)

plt.ylabel("Number of Injuries")

plt.legend(bbox_to_anchor=(0.55,0.5))

plt.xticks(rotation='vertical')

plt.xlabel(" ")



plt.subplot(1, 2, 2,).title.set_text("Injury Type")

sns.set_style("whitegrid")

sns.countplot(x ='BodyPart', hue='Surface', data=results_final,palette = sns.color_palette(color))

plt.ylabel("Number of Injuries")

plt.legend(bbox_to_anchor=(0.82,0.6))

plt.xticks(rotation=70)

plt.xlabel(" ")



plt.text(x = -6.5, y = 32.2, s = "Do Stadium Characteristics Influence Injuries?",

               fontsize = 26, color = "black", weight = 'bold', alpha = .75)

plt.text(x = -6.5, y = 30.3, 

               s = "Comparison between Field Surfaces on Player Position and Injury Type",

              fontsize = 15, color = "black", alpha = .85)

plt.text(x = -6.5, y = -15, s = 'Source: NFL 1st and Future - Playing Surface Analytics                            https://www.kaggle.com/c/NFL-playing-surface-analytics',fontsize = 14, color = 'white', backgroundcolor = 'gray')

color = ["#013369", "#D50A0A", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

color2 = ["#241075", "#BC9428", "#A5ACAF", "#66C010", "#4790DE", "#2ecc71"]



plt.figure(figsize = (15,5), facecolor = "white",)

plt.rcParams['font.size'] = 15

plt.subplot(1, 1, 1,).title.set_text("Play Type")

sns.set_style("whitegrid")

sns.countplot(x = 'PlayType', hue='Surface', data=results_final, palette = sns.color_palette(color),)

plt.ylabel("Number of Injuries")

plt.legend(bbox_to_anchor=(0.9,0.5))

plt.xticks(rotation='vertical')

plt.xlabel(" ")





plt.text(x = -.5, y = 22.2, s = "Do Stadium Characteristics Influence Injuries?",

               fontsize = 26, color = "black", weight = 'bold', alpha = .75)

plt.text(x = -.5, y = 20.8, 

               s = "Comparison between Field Surfaces on Play Type",

              fontsize = 15, color = "black", alpha = .85)

plt.text(x = -.5, y = -12, s = 'Source: NFL 1st and Future - Playing Surface Analytics                            https://www.kaggle.com/c/NFL-playing-surface-analytics',fontsize = 14, color = 'white', backgroundcolor = 'gray')

color = ["#013369", "#D50A0A", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

color2 = ["#241075", "#BC9428", "#A5ACAF", "#66C010", "#4790DE", "#2ecc71"]



plt.figure(figsize = (15,5), facecolor = "white",)

plt.rcParams['font.size'] = 15

plt.subplot(1, 1, 1,).title.set_text("NFL Season Week")

sns.set_style("whitegrid")

sns.countplot(x = 'PlayerGame', hue='Surface', data=results_final, palette = sns.color_palette(color),)

plt.ylabel("Number of Injuries")

plt.legend(bbox_to_anchor=(0.8,0.5))



plt.xlabel(" ")





plt.text(x = 0, y = 10, s = "Do Stadium Characteristics Influence Injuries?",

               fontsize = 26, color = "black", weight = 'bold', alpha = .75)

plt.text(x = 0, y = 9.5, 

               s = "Comparison between Field Surfaces on Week of Injury",

              fontsize = 15, color = "black", alpha = .85)

plt.text(x = 0, y = -1.5, s = 'Source: NFL 1st and Future - Playing Surface Analytics                            https://www.kaggle.com/c/NFL-playing-surface-analytics',fontsize = 14, color = 'white', backgroundcolor = 'gray')

color = ["#013369", "#D50A0A", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

color2 = ["#241075", "#BC9428", "#A5ACAF", "#66C010", "#4790DE", "#2ecc71"]

color3 = ["#99213E", "#FFB700", "#000000", "#66C010", "#4790DE", "#2ecc71"]



plt.figure(figsize = (15,5), facecolor = "white",)

plt.rcParams['font.size'] = 15

plt.subplot(1, 1, 1,).title.set_text("Injury Type")

sns.set_style("whitegrid")

sns.countplot(x = 'StadiumType', hue='BodyPart', data=results_final, palette = sns.color_palette(color2),)

plt.ylabel("Number of Injuries")

plt.legend(bbox_to_anchor=(0.8,0.8))



plt.xlabel(" ")





plt.text(x = -0.5, y = 38, s = "Do Stadium Characteristics Influence Injuries?",

               fontsize = 26, color = "black", weight = 'bold', alpha = .75)

plt.text(x = -0.5, y = 36.2, 

               s = "Comparison of Stadium Types on Type of Injury",

              fontsize = 15, color = "black", alpha = .85)

plt.text(x = -0.5, y = -6.5, s = 'Source: NFL 1st and Future - Playing Surface Analytics                            https://www.kaggle.com/c/NFL-playing-surface-analytics',fontsize = 14, color = 'white', backgroundcolor = 'gray')

# Question 2: Do the characteristics of the player influence injuries?



color = ["#013369", "#D50A0A", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

color2 = ["#241075", "#BC9428", "#A5ACAF", "#66C010", "#4790DE", "#2ecc71"]

color3 = ["#99213E", "#FFB700", "#000000", "#66C010", "#4790DE", "#2ecc71"]



plt.figure(figsize = (15,5), facecolor = "white",)

plt.rcParams['font.size'] = 15

plt.subplot(1, 2, 1,).title.set_text("Stadium Type")

sns.set_style("whitegrid")

sns.countplot(x = 'RosterPosition', hue='StadiumType', data=results_final, palette = sns.color_palette(color3),)

plt.ylabel("Number of Injuries")

plt.legend(bbox_to_anchor=(0.55,0.5))

plt.xticks(rotation='vertical')

plt.xlabel(" ")



plt.subplot(1, 2, 2,).title.set_text("Injury Type")

sns.set_style("whitegrid")

sns.countplot(x ='RosterPosition', hue='BodyPart', data=results_final,palette = sns.color_palette(color2))

plt.ylabel("Number of Injuries")

plt.legend(bbox_to_anchor=(0.45,0.5))

plt.xticks(rotation='vertical')

plt.xlabel(" ")



plt.text(x = -9.5, y = 16.2, s = "Do Player Characteristics Influence Injuries?",

               fontsize = 26, color = "black", weight = 'bold', alpha = .75)

plt.text(x = -9.5, y = 15.3, 

               s = "Comparison between Play Position on Stadium Types and Injury Types",

              fontsize = 15, color = "black", alpha = .85)

plt.text(x = -9.6, y = -8, s = 'Source: NFL 1st and Future - Playing Surface Analytics                            https://www.kaggle.com/c/NFL-playing-surface-analytics',fontsize = 14, color = 'white', backgroundcolor = 'gray')

# Question 3: Do the characteristics of the weather influence injuries?



color = ["#013369", "#D50A0A", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

color2 = ["#241075", "#BC9428", "#A5ACAF", "#66C010", "#4790DE", "#2ecc71"]

color3 = ["#99213E", "#FFB700", "#000000", "#66C010", "#4790DE", "#2ecc71"]



plt.figure(figsize = (15,5), facecolor = "white",)

plt.rcParams['font.size'] = 15

plt.subplot(1, 1, 1,).title.set_text("Weather")

sns.set_style("whitegrid")

sns.countplot(x = 'RosterPosition', hue='Weather', data=results_final, palette = sns.color_palette(color3),)

plt.ylabel("Number of Injuries")

plt.xticks(rotation='vertical')

plt.legend(bbox_to_anchor=(0.85,0.5))

plt.xlabel(" ")





plt.text(x = -0.5, y = 13.8, s = "Do Weather Characteristics Influence Injuries?",

               fontsize = 26, color = "black", weight = 'bold', alpha = .75)

plt.text(x = -0.5, y = 13.1, 

               s = "Comparison of Weather Types on Player Positions",

              fontsize = 15, color = "black", alpha = .85)

plt.text(x = -0.6, y = -7, s = 'Source: NFL 1st and Future - Playing Surface Analytics                            https://www.kaggle.com/c/NFL-playing-surface-analytics',fontsize = 14, color = 'white', backgroundcolor = 'gray')

color = ["#013369", "#D50A0A", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

color2 = ["#241075", "#BC9428", "#A5ACAF", "#66C010", "#4790DE", "#2ecc71"]

color3 = ["#99213E", "#FFB700", "#000000", "#66C010", "#4790DE", "#2ecc71"]



plt.figure(figsize = (15,5), facecolor = "white",)

plt.rcParams['font.size'] = 15

plt.subplot(1, 1, 1,).title.set_text("Weather")

sns.set_style("whitegrid")

sns.countplot(x = 'BodyPart', hue='Weather', data=results_final, palette = sns.color_palette(color3),)

plt.ylabel("Number of Injuries")



plt.legend(bbox_to_anchor=(0.85,0.8))

plt.xlabel(" ")





plt.text(x = -0.5, y = 23.8, s = "Do Weather Characteristics Influence Injuries?",

               fontsize = 26, color = "black", weight = 'bold', alpha = .75)

plt.text(x = -0.5, y = 22.6, 

               s = "Comparison of Weather Types on Injury Types",

              fontsize = 15, color = "black", alpha = .85)

plt.text(x = -0.6, y = -4, s = 'Source: NFL 1st and Future - Playing Surface Analytics                            https://www.kaggle.com/c/NFL-playing-surface-analytics',fontsize = 14, color = 'white', backgroundcolor = 'gray')

print(playlist['StadiumType'].unique())
print(playlist['Weather'].unique())