import pandas as pd

import numpy as np

import math 

import seaborn as sns

import matplotlib.pyplot as plt
playlist = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

playertrack= pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')

injuries= pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
injuries.head()
playlist.head()
playertrack.head()
playertrack.count()
InjuryKeys = injuries['PlayKey'].unique()

len(injuries['PlayKey'].unique())



pt_filter = playertrack['PlayKey'].isin(InjuryKeys)

playertrack2 = playertrack[pt_filter]
PlayKeys = playertrack2['PlayKey'].unique()

len(playertrack2['PlayKey'].unique())
speed = []



for i in range(0,21904):

      s = math.sqrt((round((playertrack2.iloc[i+1]['x'] - playertrack2.iloc[i]['x'])*100,3))**2 + (round((playertrack2.iloc[i+1]['y'] - playertrack2.iloc[i]['y'])*100,3))**2)

      speed.append(s)

    

speed.append(1000)

speed2 = speed



for i in range(0,21904):

    if(( playertrack2.iloc[i]['PlayKey'] != playertrack2.iloc[i+1]['PlayKey'])):

      speed2[i] = 1000



playertrack2['speed'] = speed2
dc = []



for j in range(0,21904):

      d = abs(playertrack2.iloc[j+1]['dir'] - playertrack2.iloc[j]['dir'])

      dc.append(d)



dc.append(1000)



for j in range(0,21904):

    if(( playertrack2.iloc[j]['PlayKey'] != playertrack2.iloc[j+1]['PlayKey'])):

      dc[j] = 1000

    

playertrack2['dc'] = dc
ac = []

for k in range(0,21904):

  a = playertrack2.iloc[k+1]['speed'] - playertrack2.iloc[k]['speed']

  ac.append(a)



ac.append(1000)



for k in range(0,21904):

    if(( playertrack2.iloc[k]['PlayKey'] != playertrack2.iloc[k+1]['PlayKey'])):

      ac[k-1] = 1000

      ac[k] = 1000

        

playertrack2['ac'] = ac
playertrack2.head(10)
# Use seaborn style defaults and set the default figure size

sns.set(rc={'figure.figsize':(16, 5)})
pt_sp = playertrack2[playertrack2['speed']!=1000]
# We take a look at the maximum time of game.

pt_sp['time'].max()
ax = sns.lineplot(x="time", y="speed",

             data=pt_sp)

ax.set_ylabel('Speed')

ax.set_xlabel('time')

ax.set_title("Lineplot of Speed by Time")
pt_sp2 = pd.merge(pt_sp, injuries, on = "PlayKey")
pt_sp3 = pd.merge(pt_sp2, playlist, on = "PlayKey")
pt_sp3.head()
fig, axes = plt.subplots(1, 1, figsize=(16, 10), sharex=True)

ax = sns.boxplot(data=pt_sp3, x='PlayKey', y='speed',

            whis="range", hue="Surface",hue_order =["Natural", "Synthetic"], dodge=False)

ax.set_ylabel('Speed')

ax.set_xlabel('Players')

ax.set_title("Boxplot of Speed for Injuried Players by Surface")
pt_sp2.groupby('Surface').agg({"speed":['mean','max','min','var']})
w = []



for wea in pt_sp3['Weather']:

  if (wea == "Sunny" or wea == "Clear and warm" or wea == "Clear Skies" or wea == "Clear skies" or wea == "Mostly Sunny"or wea == "Mostly sunny"or wea == "Clear"):

    w.append("Sunny")

  elif(wea == "Partly Cloudy" or wea == "Controlled Climate" or wea == "Cloudy" or wea == "Mostly cloudy" or wea == "Sun & clouds" or wea == "Coudy" or wea == "Cloudy and Cool" ):

    w.append("Cloudy")

  elif(wea == "Rain"or wea == "Light Rain"or wea == "Rain shower" or wea =="Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph." or wea =="Fair" or wea =="Cloudy, 50% change of rain" ):

    w.append("Rain")

  elif(wea == "Indoors"or wea =="Indoor"):

    w.append("Indoors")

  elif(wea == "Cold"):

    w.append("Cold")

  else:

    w.append("NA")

pt_sp3['wea'] = w
fig, axes = plt.subplots(1, 1, figsize=(16, 10), sharex=True)

ax = sns.boxplot(data=pt_sp3, x='PlayKey', y='speed',

            whis="range", hue="wea", palette = "vlag",dodge=False)

ax.set_ylabel('Speed')

ax.set_xlabel('Players')

ax.set_title("Boxplot of Speed for Injuried Players by Weather")
pt_sp_N = pt_sp3[pt_sp3['Surface']=="Natural"]

pt_sp_S = pt_sp3[pt_sp3['Surface']=="Synthetic"]
pt_sp_N['PlayKey'].unique()
p31070_3_7 = pt_sp3[pt_sp3['PlayKey'] == "31070-3-7"]

p33337_8_15 = pt_sp3[pt_sp3['PlayKey'] == "33337-8-15"]

p33474_19_7 = pt_sp3[pt_sp3['PlayKey'] == "33474-19-7"]

p34347_5_9 = pt_sp3[pt_sp3['PlayKey'] == "34347-5-9"]

p35570_15_35 = pt_sp3[pt_sp3['PlayKey'] == "35570-15-35"]

p36559_12_65 = pt_sp3[pt_sp3['PlayKey'] == "36559-12-65"]

p36621_13_58 = pt_sp3[pt_sp3['PlayKey'] == "36621-13-58"]

p38192_8_8 = pt_sp3[pt_sp3['PlayKey'] == "38192-8-8"]

p38876_29_14 = pt_sp3[pt_sp3['PlayKey'] == "38876-29-14"]

p39956_2_14 = pt_sp3[pt_sp3['PlayKey'] == "39956-2-14"]
court = plt.imread("../input/football2/football.jpg")
plt.figure(figsize=(15, 11.5))



# Plot the movemnts as scatter plot

# using a colormap to show change in game clock

plt.scatter(p31070_3_7.x, p31070_3_7.y, c=p31070_3_7.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p33337_8_15.x, p33337_8_15.y, c=p33337_8_15.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p33474_19_7.x, p33474_19_7.y, c=p33474_19_7.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p34347_5_9.x, p34347_5_9.y, c=p34347_5_9.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p35570_15_35.x, p35570_15_35.y, c=p35570_15_35.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p36559_12_65.x, p36559_12_65.y, c=p36559_12_65.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p36621_13_58.x, p36621_13_58.y, c=p36621_13_58.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p38192_8_8.x, p38192_8_8.y, c=p38192_8_8.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p38876_29_14.x, p38876_29_14.y, c=p38876_29_14.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p39956_2_14.x, p39956_2_14.y, c=p39956_2_14.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

# Darker colors represent moments earlier on in the game

cbar = plt.colorbar(orientation="horizontal")

cbar.ax.invert_xaxis()



plt.imshow(court, zorder=0, extent=[0,120,-10,60])



plt.show()
pt_sp_S['PlayKey'].unique()
p35611_7_42 = pt_sp3[pt_sp3['PlayKey'] == "35611-7-42"]

p36557_1_70 = pt_sp3[pt_sp3['PlayKey'] == "36557-1-70"]

p36607_16_19 = pt_sp3[pt_sp3['PlayKey'] == "36607-16-19"]

p38228_1_4 = pt_sp3[pt_sp3['PlayKey'] == "38228-1-4"]

p38364_5_23 = pt_sp3[pt_sp3['PlayKey'] == "38364-5-23"]

p39656_2_38 = pt_sp3[pt_sp3['PlayKey'] == "39656-2-38"]

p39678_2_1 = pt_sp3[pt_sp3['PlayKey'] == "39678-2-1"]

p39850_9_2 = pt_sp3[pt_sp3['PlayKey'] == "39850-9-2"]

p39873_4_32 = pt_sp3[pt_sp3['PlayKey'] == "39873-4-32"]

p40474_1_8 = pt_sp3[pt_sp3['PlayKey'] == "40474-1-8"]
plt.figure(figsize=(15, 11.5))



# Plot the movemnts as scatter plot

# using a colormap to show change in game clock

plt.scatter(p35611_7_42.x, p35611_7_42.y, c=p35611_7_42.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p36557_1_70.x, p36557_1_70.y, c=p36557_1_70.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p36607_16_19.x, p36607_16_19.y, c=p36607_16_19.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p38228_1_4.x, p38228_1_4.y, c=p38228_1_4.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p38364_5_23.x, p38364_5_23.y, c=p38364_5_23.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p39656_2_38.x, p39656_2_38.y, c=p39656_2_38.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p39678_2_1.x, p39678_2_1.y, c=p39678_2_1.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p39850_9_2.x, p39850_9_2.y, c=p39850_9_2.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p39873_4_32.x, p39873_4_32.y, c=p39873_4_32.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

plt.scatter(p40474_1_8.x, p40474_1_8.y, c=p40474_1_8.time,

            cmap=plt.cm.Blues, s=50, zorder=1)

# Darker colors represent moments earlier on in the game

cbar = plt.colorbar(orientation="horizontal")

cbar.ax.invert_xaxis()



plt.imshow(court, zorder=0, extent=[0,120,-10,60])



plt.show()
p_largeDC = pt_sp3[pt_sp3['dc']>=120]

p_smallDC = pt_sp3[pt_sp3['dc']<120]
p_largeDC['speed'].mean()
p_smallDC['speed'].mean()
sns.jointplot(x=pt_sp2["x"], y=pt_sp2["y"], kind='hex', marginal_kws=dict(bins=50, rug=True))
#sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='kde', color="grey", space=0)

 

# Huge space

sns.jointplot(x=pt_sp2["x"], y=pt_sp2["y"], kind='kde', color="grey", space=3)

 

# Make marginal bigger:

#sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='kde',ratio=1)

ff = playlist['PlayKey'].isin(injuries['PlayKey'])

in_list = playlist[ff]

out_list = playlist[~ff]
in_list['injuried'] = ["Y"] * 76

out_list['injuried'] = ["N"] * 266929
merged = pd.concat([in_list, out_list])

merged.head()
w2 = []



for wea in merged['Weather']:

  if (wea == "Sunny" or wea == "Clear and warm" or wea == "Clear Skies" or wea == "Clear skies" or wea == "Mostly Sunny"or wea == "Mostly sunny"or wea == "Clear"):

    w2.append(1)

  elif(wea == "Partly Cloudy" or wea == "Controlled Climate" or wea == "Cloudy" or wea == "Mostly cloudy" or wea == "Sun & clouds" or wea == "Coudy" or wea == "Cloudy and Cool" ):

    w2.append(2)

  elif(wea == "Rain"or wea == "Light Rain"or wea == "Rain shower" or wea =="Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph." or wea =="Fair" or wea =="Cloudy, 50% change of rain" ):

    w2.append(3)

  elif(wea == "Indoors"or wea =="Indoor"):

    w2.append(4)

  elif(wea == "Cold"):

    w2.append(5)

  else:

    w2.append(6)

    

merged['wea'] = w2
def hexbin(x, y, color, **kwargs):

    cmap = sns.light_palette(color, as_cmap=True)

    plt.hexbin(x, y, gridsize=10, cmap=cmap, **kwargs)



with sns.axes_style("dark"):

    g = sns.FacetGrid(merged, col="FieldType", height=4)

g.map(hexbin, "wea", "Temperature", extent=[0, 10, 0, 100]);
merged = merged[merged['Temperature']>0]
sns.catplot(x="wea", y="Temperature", hue="FieldType", kind="box", data=merged)
sns.catplot(x="FieldType", col="wea", col_wrap=6, kind="count", palette="ch:.25", data=merged)
sns.catplot(x="FieldType", col="wea", col_wrap=6, kind="count", palette="ch:.25", data=merged[merged['injuried']=="Y"])