import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from pandas import DataFrame

import matplotlib.colors as mcolors

import matplotlib.patheffects as path_effects

from matplotlib import cm

import itertools

from matplotlib.patches import Rectangle, Polygon

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image

from textwrap import wrap

import re

import warnings



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use('default')

kaggle_color = '#20beff'
playlist=pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/PlayList.csv")

injuryrecord=pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv")
# Define variable indicated number of missing training days for each injury

injuryrecord["DayMissingTraining"]=injuryrecord["DM_M1"]+injuryrecord["DM_M7"]+injuryrecord["DM_M28"]+injuryrecord["DM_M42"]

injuryrecord["DayMissingTraining"]=injuryrecord["DayMissingTraining"].replace(4,42)

injuryrecord["DayMissingTraining"]=injuryrecord["DayMissingTraining"].replace(3,28)

injuryrecord["DayMissingTraining"]=injuryrecord["DayMissingTraining"].replace(2,7)

injuryrecord["DayMissingTraining"]=injuryrecord["DayMissingTraining"].replace(1,1)



#Find number of players for each type of injury

injury1=pd.DataFrame(injuryrecord.groupby(["PlayerKey","BodyPart"])["BodyPart"].count().unstack().fillna(0))

#Find number of missing training days for each injured player

injury2=pd.DataFrame(injuryrecord.groupby(["PlayerKey"])["DayMissingTraining"].mean().fillna(0))



injury=pd.concat([injury1,injury2],axis=1)

#Find number of injury 

injury["NoInjury"]=injury.Ankle+injury.Foot+injury.Heel+injury.Knee+injury.Toes

injuryrecord["BodyPart"]=injuryrecord["BodyPart"].astype("category")

injuryrecord["BodyPart"]=injuryrecord["BodyPart"].cat.set_categories(["Knee","Ankle","Toes","Foot","Heel"])



injuryrecord["DayMissingTraining"]=injuryrecord["DayMissingTraining"].astype("category")

injuryrecord["DayMissingTraining"]=injuryrecord["DayMissingTraining"].cat.set_categories([7,1,42,28])

def draw_border_around_axes(this_ax, color="black"):

    for axis in ['top','bottom','left','right']:

        this_ax.spines[axis].set_visible(True)

        this_ax.spines[axis].set_color(color)



def hide_axes(this_ax):

    this_ax.set_frame_on(False)

    this_ax.set_xticks([])

    this_ax.set_yticks([])

    return this_ax



barstyle = {"edgecolor":"black", "linewidth":0.5}



        

f, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,9),gridspec_kw={'height_ratios':[2,5], 'width_ratios':[2,5], 'wspace':0.1, 'hspace':0.1})

this_ax = ax[0,0]

hide_axes(this_ax)



hm_ax = ax[1,1]

sns.heatmap(pd.crosstab(injuryrecord["BodyPart"],injuryrecord["DayMissingTraining"]),annot=True, fmt="d", square=False, 

center = 90, vmin=0, vmax=20, lw=4, cbar=False,color="black")

hm_ax.yaxis.tick_right()

hm_ax.yaxis.set_label_position("right")

draw_border_around_axes(hm_ax)



this_ax = ax[1,0]

injuryrecord.BodyPart.value_counts().to_frame().sort_values(by="BodyPart").plot.barh(ax=this_ax,colors='darkseagreen',**barstyle)

this_ax.set_xlabel("BodyPart")

this_ax.xaxis.tick_top()

this_ax.set_xlim(this_ax.get_xlim()[::-1]);

this_ax.yaxis.set_label_position("right")

this_ax.xaxis.set_label_position("top")





this_ax = ax[0,1]

injuryrecord.DayMissingTraining.value_counts().plot.bar(ax=this_ax,colors='peru',**barstyle)

this_ax.set_ylabel("MissingDays")

this_ax.xaxis.set_label_position("bottom")

this_ax.yaxis.set_label_position("left")

this_ax.xaxis.tick_top()

injuryrecord.groupby(['BodyPart','Surface']).count().unstack('BodyPart')['PlayerKey'].T.apply(lambda x: x).sort_values('BodyPart').T.sort_values("Knee", ascending=False).plot(kind='barh',

          figsize=(15, 5),

          title='Injury Body Part by Field Type',

          stacked=True)

plt.show()

# Data cleaning

#1-Stadium Type

playlist.StadiumType=playlist.StadiumType.replace(["Outdoor","Outdoors","Oudoor","Ourdoor","Outdoor Retr Roof-Open","Outddors",

                                                  "Outdor","Outside","Open","cloudy","Cloudy","Bowl","Heinz Field","Indoor, Open Roof"

                                                  ,"Domed, open","Domed, Open","Retr. Roof - Open","Retr. Roof-Open","Open"],"Outdoor")

playlist.StadiumType=playlist.StadiumType.replace(["Indoors","Indoor","Indoor, Roof Closed","Dome","Domed, closed","Dome, closed","Closed Dome",

                                                   "Domed","Retractable Roof","Retr. Roof - Closed","Retr. Roof-Closed","Retr. Roof Closed"],"Indoor")

#2-Weather

playlist.Weather=playlist.Weather.replace(["Cloudy","Partly Cloudy","Mostly Cloudy","Partly Clouidy","Coudy",

                                           "Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.",

                                          "Party Cloudy","Cloudy, chance of rain","Mostly cloudy","Partly cloudy","cloudy","Cloudy and Cool"

                                          "Cloudy, 50% change of rain","Cloudy, fog started developing in 2nd quarter",

                                           "Cloudy, 50% change of rain","Cloudy and Cool","Cloudy and cold","Cloudy, light snow accumulating 1-3",

                                          "Mostly Coudy","Cloudy, light snow accumulating 1-3","Hazy","Overcast"],"Cloudy")

playlist.Weather=playlist.Weather.replace(["Sunny","Mostly Sunny","Partly Sunny","Mostly sunny","Sunny and clear",

                                           "Sunny and warm","Sunny, highs to upper 80s","Sunny Skies","Sunny, Windy","Partly sunny",

                                           "Clear and Sunny","Sun & clouds","Clear and sunny","Mostly Sunny Skies","Sunny and cold"],"Sunny")

playlist.Weather=playlist.Weather.replace(["Clear","Clear and warm","Clear Skies","Clear skies","Clear and cold","Partly clear",

                                           "Clear and Cool","Clear to Partly Cloudy","Fair"],"Clear")

playlist.Weather=playlist.Weather.replace(["N/A (Indoors)","Indoors","Indoor","N/A Indoor","Controlled Climate"],"Indoor")

playlist.Weather=playlist.Weather.replace(["Light Rain","Scattered Showers","Showers","Rainy","Rain shower","Cloudy, Rain",

                                           "Chance of Rain","Rain likely, temps in low 40s.","10% Chance of Rain","Rain Chance 40%",

                                          "30% Chance of Rain"],"Rain")

playlist.Weather=playlist.Weather.replace(["Heavy lake effect snow","Snow",'Cloudy, light snow accumulating 1-3"'],"Snow")

playlist.Weather=playlist.Weather.replace(["Cold","Heat Index 95"],"Other")



#3-RosterPosition

playlist.RosterPosition=playlist.RosterPosition.replace(["Offensive Lineman"],"Offensive_Lineman")

playlist.RosterPosition=playlist.RosterPosition.replace(["Wide Receiver"],"Wide_Receiver")

playlist.RosterPosition=playlist.RosterPosition.replace(["Defensive Lineman"],"Defensive_Lineman")

playlist.RosterPosition=playlist.RosterPosition.replace(["Running Back"],"Running_Back")

playlist.RosterPosition=playlist.RosterPosition.replace(["Tight End"],"Tight_End")



#4-PlayType

playlist.PlayType=playlist.PlayType.replace(["0"],"None")



#5-playlist

playlist.PlayType=playlist.PlayType.replace(["Extra Point"],"Extra-Point")

playlist.PlayType=playlist.PlayType.replace(["Field Goal"],"Field-Goal")

playlist.PlayType=playlist.PlayType.replace(["Kickoff Not Returned"],"Kickoff-Not-Returned")

playlist.PlayType=playlist.PlayType.replace(["Kickoff Returned"],"Kickoff-Returned")

playlist.PlayType=playlist.PlayType.replace(["Punt Not Returned"],"Punt-Not-Returned")

playlist.PlayType=playlist.PlayType.replace(["Punt Returned"],"Punt-Returned")



# Missing data

#stadium type

playlist["StadiumType"].fillna("Outdoor",inplace=True)

# Weather and Temerature

playlist.Temperature.replace([-999],np.nan,inplace=True)

playlist["Weather"]=playlist.groupby("StadiumType").Weather.transform(lambda x: x.fillna(x.mode()[0]))

playlist["Temperature"].fillna(playlist.groupby(["StadiumType","Weather"])["Temperature"].transform(np.median),inplace=True)

#split the playlist datasets into two sets one for inured players and other for non injured players

Injplaylist=playlist[playlist.PlayerKey.isin(injuryrecord.PlayerKey)]

Injplay=Injplaylist.dropna(axis=0)

play_noinjury=playlist[~playlist.PlayerKey.isin(injuryrecord.PlayerKey)]

play_noinjury.dropna(axis=0,inplace=True)

play_injury=pd.merge(Injplaylist,injuryrecord,on=["PlayerKey","GameID","PlayKey"],how="outer")

play_injury=play_injury.drop(["DM_M1","DM_M7","DM_M28","DM_M42","DayMissingTraining"],axis=1)

play_injury["BodyPart"]=play_injury["BodyPart"].astype("category")

play_injury["BodyPart"]=play_injury["BodyPart"].cat.add_categories('None')

play_injury["BodyPart"].fillna("None",inplace=True)

play_injury.dropna(axis=0,inplace=True)
play_injury["Injury_Risk"]=1

play_noinjury["Injury_Risk"]=0

injuryrisk=pd.concat([play_injury,play_noinjury],axis=0)

injuryrisk.drop(["BodyPart"],axis=1,inplace=True)



risk1=injuryrisk.groupby(["PlayerKey","RosterPosition"])["RosterPosition"].nunique().unstack().fillna(0).astype('int64')

risk4=injuryrisk.groupby(["PlayerKey","PlayType"])["PlayType"].count().unstack().fillna(0).astype('int64')

risk6=injuryrisk.groupby(["PlayerKey"])["PlayerGame"].max().fillna(0).astype('int64')

risk7=injuryrisk.groupby(["PlayerKey"])["PlayerGamePlay"].max().fillna(0).astype('int64')

risk8=injuryrisk.groupby(["PlayerKey"])["Temperature"].max().fillna(0).astype('int64')

risk9=injuryrisk.groupby(["PlayerKey","FieldType"])["FieldType"].count().unstack().fillna(0).astype('int64')

risk10=injuryrisk.groupby(["PlayerKey","StadiumType"])["StadiumType"].count().unstack().fillna(0).astype('int64')

risk11=injuryrisk.groupby(["PlayerKey","Weather"])["Weather"].count().unstack().fillna(0).astype('int64')

risk12=injuryrisk.groupby(["PlayerKey"])["Injury_Risk"].mean().fillna(0).astype('int64')



from functools import reduce

dfs = [risk1, risk4,risk6, risk7,risk8, risk9, risk10, risk11, risk12] # list of dataframes

df_final = reduce(lambda left,right: pd.merge(left,right,on='PlayerKey'), dfs)



plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

text1=" ".join(Injplay['RosterPosition'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white', max_words=300,collocations = False).generate(text1)

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Roster Position game play distribution for player with injury', fontsize=15)

plt.axis("off")



plt.subplot(1,2,2)

text2=" ".join(play_noinjury['RosterPosition'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white', max_words=300,collocations = False).generate(text2)

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Roster Position game play distribution for player with no injury', fontsize=15)

plt.axis("off")

plt.show()

import statsmodels.api as sm

x=df_final[['Cornerback', 'Defensive_Lineman', 'Kicker', 'Linebacker','Offensive_Lineman', 'Quarterback', 'Running_Back', 'Safety','Tight_End', 'Wide_Receiver']]

y=df_final.Injury_Risk

sm.Logit(y,x).fit().pvalues
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

text1=" ".join(Injplay['PlayType'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white', max_words=300,collocations = False).generate(text1)

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Play Type game play distribution for player with injury')

plt.axis("off")



plt.subplot(1,2,2)

text2=" ".join(play_noinjury['PlayType'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white', max_words=300,collocations = False).generate(text2)

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Play Type  game play distribution for player with no injury')

plt.axis("off")



plt.show()



x=df_final[['Extra-Point', 'Field-Goal', 'Kickoff','Kickoff-Not-Returned', 'Kickoff-Returned', 'None', 'Pass', 'Punt',

           'Punt-Not-Returned', 'Punt-Returned', 'Rush']]

y=df_final.Injury_Risk

sm.Logit(y,x).fit().pvalues
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

text1=" ".join(Injplay['Weather'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white', max_words=300,collocations = False).generate(text1)

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Weather distribution for player with injury',fontsize=15)

plt.axis("off")



plt.subplot(1,2,2)

text2=" ".join(play_noinjury['Weather'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white', max_words=300,collocations = False).generate(text2)

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Weather distribution for player with no injury',fontsize=15)

plt.axis("off")

plt.show()

x=df_final[['Clear','Cloudy', 'Indoor_y', 'Other', 'Rain', 'Snow', 'Sunny']]

y=df_final.Injury_Risk

sm.Logit(y,x).fit().pvalues
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

ax1=Injplaylist.FieldType.value_counts().plot.barh(color="darkseagreen")

plt.gca().invert_xaxis()

plt.title("Field Type game play distribution for injury players",fontsize=15)



plt.subplot(1,2,2)

ax2=play_noinjury.FieldType.value_counts().plot.barh(color="peru")

plt.title("Field Type game play distribution for non_injury players",fontsize=15)



plt.subplots_adjust(wspace=0.05)
x=df_final[['Natural', 'Synthetic']]

y=df_final.Injury_Risk

sm.Logit(y,x).fit().pvalues
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

ax1=Injplaylist.StadiumType.value_counts().plot.barh(color="darkseagreen")

plt.gca().invert_xaxis()

plt.title("Stadium Type game play distribution for injurt players",fontsize=15)



plt.subplot(1,2,2)

ax2=play_noinjury.StadiumType.value_counts().plot.barh(color="peru")

plt.title("Stadium Type game play distribution for non_injurt players",fontsize=15)



plt.subplots_adjust(wspace=0.05)
x=df_final[["Indoor_x","Outdoor"]]

y=df_final.Injury_Risk

sm.Logit(y,x).fit().pvalues
Nogame_injury=play_injury.groupby(["PlayerKey"])["PlayerGame"].max().to_frame()

Nogame_injury.columns=["No.game.inj"]

Nogame_noinjury=play_noinjury.groupby(["PlayerKey"])["PlayerGame"].max().to_frame()

Nogame_noinjury.columns=["No.game.noinj"]



plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

ax1=Nogame_injury["No.game.inj"].hist(color="darkseagreen")

plt.xlabel("Number of game")

plt.title("Number of games played by injured player",fontsize=15)



ax2=plt.subplot(1,2,2)

Nogame_noinjury["No.game.noinj"].hist(color="peru")

plt.xlabel("Number of game")

plt.title("Number of games played by non_injured player",fontsize=15)



plt.show()

x=df_final[["PlayerGame"]]

y=df_final.Injury_Risk

sm.Logit(y,x).fit().pvalues
Nogameplay_injury=play_injury.groupby(["PlayerKey"])["PlayerGamePlay"].max().to_frame()

Nogameplay_injury.columns=["No.gameplay.inj"]

Nogameplay_noinjury=play_noinjury.groupby(["PlayerKey"])["PlayerGamePlay"].max().to_frame()

Nogameplay_noinjury.columns=["No.gameplay.noinj"]



plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

Nogameplay_injury["No.gameplay.inj"].hist(color="darkseagreen")

plt.xlabel("Maximum number of plays per game")

plt.title("Maximum number of plays per game played by injured player",fontsize=15)





plt.subplot(1,2,2)

Nogameplay_noinjury["No.gameplay.noinj"].hist(color="peru")

plt.xlabel("Maximum number of plays per game")

plt.title("Maximum number of plays per game played by non_injured player",fontsize=15)



plt.show()
x=df_final["PlayerGamePlay"]

y=df_final.Injury_Risk

sm.Logit(y,x).fit().pvalues
Tem_injury=play_injury.groupby(["PlayerKey"])["Temperature"].mean().to_frame()

Tem_injury.columns=["tem.mean.inj"]

Tem_noinjury=play_noinjury.groupby(["PlayerKey"])["Temperature"].mean().to_frame()

Tem_noinjury.columns=["tem.mean.noinj"]



plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

Tem_injury["tem.mean.inj"].hist(color="darkseagreen")

plt.xlabel("Average Temperature")

plt.title("Temperature distribution in games played by injured players",fontsize=15)



plt.subplot(1,2,2)

Tem_noinjury["tem.mean.noinj"].hist(color="peru")

plt.xlabel("Average Temperature")

plt.title("Temperature distribution in games played by non_injured players",fontsize=15)

plt.show()
x=df_final["Temperature"]

y=df_final.Injury_Risk

sm.Logit(y,x).fit().pvalues
import dask.dataframe as dd

track = dd.read_csv("/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv")

track["PlayerKey"]=track["PlayKey"].str[:5]



playtrack=track.groupby(["PlayKey"]).agg({"x":"mean","y":"mean","dir":"mean","dis":"mean","o":"mean","s":"mean"}).compute()

play_track=pd.merge(playtrack,playlist,on="PlayKey")



sns.catplot(y="FieldType",x="x",data=play_track,kind="boxen")



sns.catplot(y="FieldType",x="y",data=play_track,kind="boxen")

plt.show()



sns.catplot(y="FieldType",x="dir",data=play_track,kind="boxen")



sns.catplot(y="FieldType",x="dis",data=play_track,kind="boxen")
sns.catplot(y="FieldType",x="o",data=play_track,kind="boxen")



sns.catplot(y="FieldType",x="s",data=play_track,kind="boxen")
playertrack=track.groupby(["PlayerKey"]).agg({"x":"mean","y":"mean","dir":"mean","dis":"mean","o":"mean","s":"mean"}).compute()

playertrack.reset_index(inplace=True)

df_final.reset_index(inplace=True)

df_final.PlayerKey = df_final.PlayerKey.astype("object")

player_track=pd.concat([df_final["Injury_Risk"],playertrack],axis=1)

player_track.dropna(axis=0,inplace=True)
x = player_track.drop(["Injury_Risk","PlayerKey"],axis=1)

y = player_track.Injury_Risk

sm.Logit(y,x).fit().pvalues