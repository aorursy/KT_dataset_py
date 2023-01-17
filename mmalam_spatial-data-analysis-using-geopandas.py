import pandas as pd

import numpy as np

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

from shapely.geometry import Point, Polygon

import matplotlib.gridspec as gridspec

import geopandas as gpd

from shapely.geometry import Point, Polygon

from mpl_toolkits.axes_grid1 import make_axes_locatable
## ALTER TABLE Matches

## ADD weight_class_2 VARCHAR(50)

## 

## ALTER TABLE Matches

## ADD Gender VARCHAR(10)

## 

## ALTER TABLE Matches

## ADD Country VARCHAR(100)

## 

## ALTER TABLE Matches

## ADD City VARCHAR(100)

## 

## ALTER TABLE Matches

## ADD Lat DECIMAL(19,9)

## 

## ALTER TABLE Matches

## ADD Lon DECIMAL(19,9)

## 

## SELECT right(location, charindex(',', reverse(location) + ',') - 1)

## FROM Matches

## 

## UPDATE Matches

## SET Country = right(location, charindex(',', reverse(location) + ',') - 1)

## 

## UPDATE Matches

## SET City = LEFT(location, CHARINDEX(', ', location)-1)

## 

## UPDATE Matches

## SET weight_class_2 = REPLACE(weight_class, 'Women''s ', '')

## 

## UPDATE Matches

## SET Gender = CASE WHEN weight_class IN ('Women''s Flyweight', 'Women''s Bantamweight', 'Women''s Strawweight') THEN 'Female' ELSE 'Male' END

## 

## UPDATE Matches

## SET Country = LTRIM(Country)

## 

## UPDATE Matches

## SET Country = 'United States'

## WHERE Country='USA'

## 

## UPDATE Matches

## SET Country = 'Korea'

## WHERE Country='South Korea'

## 

## UPDATE Matches

## SET Country = 'Czech Rep.'

## WHERE Country='Czech Republic'

## SELECT * FROM Matches
data = pd.read_csv('../input/ufc-spatial-analysis/matches.csv')

data.head()
data2 = pd.read_csv('../input/ufc-spatial-analysis/matches2.csv')

data2.head()
fig,ax = plt.subplots(figsize = (20,20))

title = plt.title('Number of Matches by Location', fontsize=20)

title.set_position([0.5, 1.05])

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world.plot(ax = ax, color='grey', edgecolor='black',linewidth=1, alpha=0.1)

sns.scatterplot(data2.Lon,data2.Lat,size=data2.matches_count, ax=ax)
# Since the data that has the matches rocords have name of "United Stated" instead of "United states of America" which is in GeoPandas

# World Map, so changing it to "United States"

world.loc[4, 'name'] = 'United States'
## SELECT Country AS [name], COUNT(*) As [matches_count]

## FROM Matches

## GROUP BY Country

## ORDER BY COUNT(*) DESC
data3 = pd.read_csv('../input/ufc-spatial-analysis/matches3.csv')

data3.head()
world_matches = world.merge(data3, on='name', how='left')

world_matches["matches_count"].fillna(0, inplace=True)

world_matches.head()
fig,ax = plt.subplots(figsize = (20,10))

title = plt.title('Number of Matches by Country', fontsize=20)

title.set_position([0.5, 1.05])

world_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax

                  ,vmin=0

                  ,edgecolor='black',linewidth=0.1)

ax.axes.get_xaxis().set_visible(False)

ax.axes.get_yaxis().set_visible(False)
## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

states = gpd.read_file('../input/us-border-crossing-temporal-and-spatial-analysis/states.shp')

data4 = pd.read_csv('../input/ufc-spatial-analysis/matches4.csv')
state_matches = states.merge(data4, on='STATE_NAME', how='left')

state_matches["matches_count"].fillna(0, inplace=True)

state_matches.head()
fig,ax = plt.subplots(figsize = (20,7))

title = plt.title('Number of Matches by State', fontsize=20)

title.set_position([0.5, 1.05])

state_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax

                 , vmax=400,vmin=0

                  ,edgecolor='black',linewidth=0.1)

ax.axes.get_xaxis().set_visible(False)

ax.axes.get_yaxis().set_visible(False)
## SELECT Country AS [name], COUNT(*) As [matches_count]

## FROM Matches

## WHERE title_bout = 'True'

## GROUP BY Country

## ORDER BY COUNT(*) DESC



## SELECT Country AS [name], COUNT(*) As [matches_count]

## FROM Matches

## WHERE title_bout = 'False'

## GROUP BY Country

## ORDER BY COUNT(*) DESC





data5_world_title =  pd.read_csv('../input/ufc-spatial-analysis/matches5_world_title.csv')

data5_world_nontitle =  pd.read_csv('../input/ufc-spatial-analysis/matches5_world_nontitle.csv')
world_title_matches = world.merge(data5_world_title, on='name', how='left')

world_title_matches["matches_count"].fillna(0, inplace=True)

world_nontitle_matches = world.merge(data5_world_nontitle, on='name', how='left')

world_nontitle_matches["matches_count"].fillna(0, inplace=True)





f,ax=plt.subplots(1,2,figsize=(20,5))



world_title_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=30,vmin=0

                  ,edgecolor='black',linewidth=0.1)

world_nontitle_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=450,vmin=0

                  ,edgecolor='black',linewidth=0.1)



f.suptitle('Matches in Country by Title', fontsize=14)

ax[0].set_title('Title Matches by Country', fontsize=12)

ax[1].set_title('Non-Title Matches by Country', fontsize=12)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND title_bout = 'True'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC



## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND title_bout = 'False'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC



data5_state_title =  pd.read_csv('../input/ufc-spatial-analysis/matches5_state_title.csv')

data5_state_nontitle =  pd.read_csv('../input/ufc-spatial-analysis/matches5_state_nontitle.csv')
state_title_matches = states.merge(data5_state_title, on='STATE_NAME', how='left')

state_title_matches["matches_count"].fillna(0, inplace=True)

state_nontitle_matches = states.merge(data5_state_nontitle, on='STATE_NAME', how='left')

state_nontitle_matches["matches_count"].fillna(0, inplace=True)



f,ax=plt.subplots(1,2,figsize=(20,5))

state_title_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=29,vmin=0

                  ,edgecolor='black',linewidth=0.1)

state_nontitle_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=335,vmin=0

                  ,edgecolor='black',linewidth=0.1)

f.suptitle('Matches in USA States by Title', fontsize=14)

ax[0].set_title('Title Matches by State', fontsize=12)

ax[1].set_title('Non-Title Matches by State', fontsize=12)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
## SELECT Country AS [name], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Gender = 'Male'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Gender = 'Female'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND Gender = 'Male'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND Gender = 'Female'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC



data6_world_male =  pd.read_csv('../input/ufc-spatial-analysis/matches6_world_male.csv')

data6_world_female =  pd.read_csv('../input/ufc-spatial-analysis/matches6_world_female.csv')

data6_state_male =  pd.read_csv('../input/ufc-spatial-analysis/matches6_state_male.csv')

data6_state_female =  pd.read_csv('../input/ufc-spatial-analysis/matches6_state_female.csv')



world_male_matches = world.merge(data6_world_male, on='name', how='left')

world_male_matches["matches_count"].fillna(0, inplace=True)

world_female_matches = world.merge(data6_world_female, on='name', how='left')

world_female_matches["matches_count"].fillna(0, inplace=True)

state_male_matches = states.merge(data6_state_male, on='STATE_NAME', how='left')

state_male_matches["matches_count"].fillna(0, inplace=True)

state_female_matches = states.merge(data6_state_female, on='STATE_NAME', how='left')

state_female_matches["matches_count"].fillna(0, inplace=True)
f,ax=plt.subplots(1,2,figsize=(20,5))



world_male_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=456,vmin=0

                  ,edgecolor='black',linewidth=0.1)

world_female_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=30,vmin=0

                  ,edgecolor='black',linewidth=0.1)



f.suptitle('Matches in Country by Gender', fontsize=14)

ax[0].set_title('Male', fontsize=12)

ax[1].set_title('Female', fontsize=12)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
f,ax=plt.subplots(1,2,figsize=(20,5))

state_male_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=347,vmin=0

                  ,edgecolor='black',linewidth=0.1)

state_female_matches.plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=20,vmin=0

                  ,edgecolor='black',linewidth=0.1)

f.suptitle('Matches in USA States by Gender', fontsize=14)

ax[0].set_title('Male', fontsize=12)

ax[1].set_title('Female', fontsize=12)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
## SELECT Country AS [name], COUNT(*) As [matches_count] 

## FROM Matches

## WHERE weight_class_2 = 'Catch Weight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count] 

## FROM Matches

## WHERE weight_class_2 = 'Light Heavyweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count] 

## FROM Matches

## WHERE weight_class_2 = 'Lightweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count] 

## FROM Matches

## WHERE weight_class_2 = 'Open Weight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count]

## FROM Matches

## WHERE weight_class_2 = 'Strawweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count] 

## FROM Matches

## WHERE weight_class_2 = 'Bantamweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count]

## FROM Matches

## WHERE weight_class_2 = 'Heavyweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count] 

## FROM Matches

## WHERE weight_class_2 = 'Featherweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count] 

## FROM Matches

## WHERE weight_class_2 = 'Welterweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count] 

## FROM Matches

## WHERE weight_class_2 = 'Middleweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

## 

## SELECT Country AS [name], COUNT(*) As [matches_count]

## FROM Matches

## WHERE weight_class_2 = 'Flyweight'

## GROUP BY Country

## ORDER BY COUNT(*) DESC

data7_world_wc = []

world_wc = []

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc1.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc2.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc3.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc4.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc5.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc6.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc7.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc8.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc9.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc10.csv'))

data7_world_wc.append(pd.read_csv('../input/ufc-spatial-analysis/matches_world_wc11.csv'))



data7_world_wc_attr = []

data7_world_wc_attr.append({"vmin": 3, "category": 'Catch Weight'})

data7_world_wc_attr.append({"vmin": 41, "category": 'Light Heavyweight'})

data7_world_wc_attr.append({"vmin": 67, "category": 'Lightweight'})

data7_world_wc_attr.append({"vmin": 6, "category": 'Open Weight'})

data7_world_wc_attr.append({"vmin": 12, "category": 'Strawweight'})

data7_world_wc_attr.append({"vmin": 49, "category": 'Bantamweight'})

data7_world_wc_attr.append({"vmin": 26, "category": 'Heavyweight'})

data7_world_wc_attr.append({"vmin": 53, "category": 'Featherweight'})

data7_world_wc_attr.append({"vmin": 77, "category": 'Welterweight'})

data7_world_wc_attr.append({"vmin": 60, "category": 'Middleweight'})

data7_world_wc_attr.append({"vmin": 21, "category": 'Flyweight'})



loop = 0

for data7 in data7_world_wc:

    world_wc.append(world.merge(data7_world_wc[loop], on='name', how='left'))

    world_wc[loop]["matches_count"].fillna(0, inplace=True)

    loop += 1
f,ax=plt.subplots(6,2,figsize=(20,25))

f.suptitle('Matches in Country by Category', fontsize=14)



row = 0

column = 0

while row < 6:

    column = 0

    while column < 2: 

        if row == 5 and column == 1:

            column += 1

            break

        world_wc[row*2+column].plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[row][column], vmax=data7_world_wc_attr[row*2+column]["vmin"],vmin=0

                  ,edgecolor='black',linewidth=0.1)

        ax[row][column].set_title(data7_world_wc_attr[row*2+column]["category"], fontsize=12)

        ax[row][column].axes.get_xaxis().set_visible(False)

        ax[row][column].axes.get_yaxis().set_visible(False)

        column += 1

    row += 1

ax[5][1].remove()
## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Catch Weight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Light Heavyweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Lightweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Open Weight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Strawweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Bantamweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Heavyweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Featherweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Welterweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Middleweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC

## 

## SELECT [State] AS [STATE_NAME], COUNT(*) As [matches_count]

## FROM Matches

## WHERE Country = 'United States' AND weight_class_2 = 'Flyweight'

## GROUP BY [State]

## ORDER BY COUNT(*) DESC



data8_state_matches = []

state_matches = []

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_1.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_2.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_3.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_4.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_5.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_6.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_7.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_8.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_9.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_10.csv'))

data8_state_matches.append(pd.read_csv('../input/ufc-spatial-analysis/matches_state_11.csv'))



loop = 0

for data8 in data8_state_matches:

    state_matches.append(states.merge(data8_state_matches[loop], on='STATE_NAME', how='left'))

    state_matches[loop]["matches_count"].fillna(0, inplace=True)

    loop += 1
f,ax=plt.subplots(6,2,figsize=(20,25))

f.suptitle('Matches in States by Category', fontsize=14)



row = 0

column = 0

while row < 6:

    column = 0

    while column < 2: 

        if row == 5 and column == 1:

            column += 1

            break

        state_matches[row*2+column].plot( column='matches_count', 

                  cmap='OrRd', legend=True, ax=ax[row][column]

                  ,edgecolor='black',linewidth=0.1)

        ax[row][column].set_title(data7_world_wc_attr[row*2+column]["category"], fontsize=12)

        ax[row][column].axes.get_xaxis().set_visible(False)

        ax[row][column].axes.get_yaxis().set_visible(False)

        column += 1

    row += 1

ax[5][1].remove()