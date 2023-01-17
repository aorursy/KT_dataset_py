#Importering av moduler och paket, med fördefinierade funktioner

import string

import re



#Datamanipulation

import numpy as np

import pandas as pd



#Data visualization

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sns



from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype

import plotly as py

import plotly.graph_objs as go



#display properties

pd.set_option("display.max_columns", 100)

pd.set_option("display.max_rows", 100)



#Date

import datetime





from math import pi

import os
#Läs in fil med FIFA-data



def seasons (start, end):

    start = int(start)

    end = int(end)

    seasons = []

    while start < end:

        if start < 10:

            seasons.append("0"+str(start)+str((int(start)+1)))

            start = (int(start)+1)

        else:

             seasons.append(str(start)+str((int(start)+1)))

             start = (int(start)+1)

    dfs = []

    for season in seasons:

        importstring = r"../input/season-"+str(season)+"_csv.csv"

        df = pd.read_csv(importstring, sep=",", engine="python")

        df["Season"] = season

        dfs.append(df)

    df = pd.concat(dfs, sort=True)

    return df

        

df = seasons("1993", "2019")
#----------------------------------- Cleaning original data

#One instance of Middlesbrough written Middlesboro

df = df.replace({"Middlesboro": "Middlesbrough"}, regex=True)



#print(df.info())

#There is missing alot of values in a few of the odds-columns, so we can drop all that are missing 100+, and div because it"s all div 1



df = df.drop(["Div", "BSA", "BSD", "BSH", "GBA", "GBD", "GBH", "LBA",

              "LBD","LBH", "PSA", "PSCA", "PSCD", 

              "PSCH", "PSD", "PSH", "SBA", "SBD", "SBH", "SJA", "SJD", "SJH"], axis=1)





#Only missing one entry (from -09) in BWA, BWD, BWH, IWA, IWD, IWH and only missing 10 entrys in BbA (unt so weiter)

#Fill the missing values with the mean for each column for now -> The smarter choice is probably to just drop them since they will probably 

    # only be used for creating a maching learning model later on.

    



#print(df.columns)

#----------------Renaming existing columns

df = df[["Season","Date", "HomeTeam", "AwayTeam", "Referee", 

         "FTAG", "FTHG", "FTR", "HTAG", "HTHG", "HTR", "HC", "HF", "HR", "HS", "HST",

       "HY",  "AC", "AF", "AR", "AS", "AST", "AY", 

       "B365A", "B365D",

       "B365H", "BWA", "BWD", "BWH", "Bb1X2", "BbAH", "BbAHh", "BbAv<2.5",

       "BbAv>2.5", "BbAvA", "BbAvAHA", "BbAvAHH", "BbAvD", "BbAvH", "BbMx<2.5",

       "BbMx>2.5", "BbMxA", "BbMxAHA", "BbMxAHH", "BbMxD", "BbMxH", "BbOU",

        "IWA", "IWD", "IWH", "VCA", "VCD", "VCH", "WHA", "WHD", "WHH"]]

dfHeaders = ["Season","Date", "HomeTeam", "AwayTeam", "Referee", 

         "FT_AwayGoals", "FT_HomeGoals", "FT_Result", "HT_AwayGoals", "HT_HomeGoals", "HT_Result", 

         "HomeCorners", "HomeFouls", "HomeRedcards", "HomeShots", "HomeShotsOnTarget",

        "HomeYellowcards",  "AwayCorners", "AwayFouls", "AwayRedcards",

       "AwayShots", "AwayShotsOnTarget", "AwayYellowcards", 

       "B365A", "B365D",

       "B365H", "BWA", "BWD", "BWH", "Bb1X2", "BbAH", "BbAHh", "BbAv<2.5",

       "BbAv>2.5", "BbAvA", "BbAvAHA", "BbAvAHH", "BbAvD", "BbAvH", "BbMx<2.5",

       "BbMx>2.5", "BbMxA", "BbMxAHA", "BbMxAHH", "BbMxD", "BbMxH", "BbOU",

        "IWA", "IWD", "IWH", "VCA", "VCD", "VCH", "WHA", "WHD", "WHH"]

df.columns = dfHeaders

del dfHeaders



#-----------------------Create columns with 1's 0 zeros for the result

df["FT_HomeWin"] = df["FT_Result"].apply(lambda result: 1 if result == "H" else  0)

df["HT_HomeWin"] = df["HT_Result"].apply(lambda result: 1 if result == "H" else 0)

df["FT_HomeDraw"] = df["FT_Result"].apply(lambda result: 1 if result == "D" else 0)

df["HT_HomeDraw"] = df["HT_Result"].apply(lambda result: 1 if result == "D" else 0)

df["FT_HomeLoss"] = df["FT_Result"].apply(lambda result: 1 if result == "A" else 0)

df["HT_HomeLoss"] = df["HT_Result"].apply(lambda result: 1 if result == "A" else 0)

df["FT_AwayWin"] = df["FT_Result"].apply(lambda result: 1 if result == "A" else 0)

df["HT_AwayWin"] = df["HT_Result"].apply(lambda result: 1 if result == "A" else 0)

df["FT_AwayDraw"] = df["FT_Result"].apply(lambda result: 1 if result == "D" else 0)

df["HT_AwayDraw"] = df["HT_Result"].apply(lambda result: 1 if result == "D" else 0)

df["FT_AwayLoss"] = df["FT_Result"].apply(lambda result: 1 if result == "H" else 0)

df["HT_AwayLoss"] = df["HT_Result"].apply(lambda result: 1 if result == "H" else 0)

df["FT_HomeConceded"] = df["FT_AwayGoals"]

df["HT_HomeConceded"] = df["HT_AwayGoals"]

df["FT_AwayConceded"] = df["FT_HomeGoals"]

df["HT_AwayConceded"] = df["HT_HomeGoals"]



#-------Creating dates that are easier to use.--------------

A = df["Date"].str.split("-", expand = True)

AHeaders = ["Year", "Month", "Day"]

A.columns = AHeaders

A = A[A["Year"].str.len() < 5]

B = df["Date"].str.split("/", expand = True)

B = B[[2, 1, 0]]

B.columns = AHeaders

B = B[B["Day"].str.len() < 5]

AB = [A, B]

A = pd.concat(AB, sort=True)

A["Year"] = A["Year"].apply(lambda year: year if len(year) > 3 

                                  else "19"+year if int(year) > 92

                                  else "20"+year)

#A["Year"] = A["Year"].str[-2:]

#Ändra det här så att det är 4 siffror i år

A = A.sort_values(["Year", "Month", "Day"])

df = pd.concat([df, A], axis=1)

df = df.drop("Date", axis=1)



del A

del B

del AHeaders





#------------Fix the Referee column to only include last name.

A = df["Referee"].str.split(" ", expand = True)

for col in A.columns:

    A[col] = A[col].apply(lambda col: col if len(str(col)) > 2 else np.nan)

    

for col in A.columns:

    A[col] = A[col].apply(lambda col: col if col != None else np.nan)

    

A[1] = A[1].fillna(A[2])

del A[2]

A[1] = A[1].fillna(A[0])

df["Referee"] = A[1]

del A

del col



df["Referee"] = df["Referee"].apply(lambda name: "Elleray" if (name == "Ellaray") 

                                  else "Gallagher" if (name == "Gallagh") or (name == "Gallaghe")

                                  else name)
#----------------------Some lists to not where there is missing data-------------------



NoHTScore = ['19931994', '19941995']

NoDeepStats = ['19931994', '19941995', '19951996', '19961997', '19971998', '19981999',

 '19992000']



#---------------------------------------------------------

#-----Create new dataframe for full seasons--------------



def TableDataFrame(df): 

   HomeHeaders = ["FT_HomeWin", "HT_HomeWin", "FT_HomeDraw", "HT_HomeDraw",

           "FT_HomeLoss", "HT_HomeLoss", "FT_HomeGoals", "HT_HomeGoals", "FT_HomeConceded", "HT_HomeConceded", 

              "HomeCorners", "HomeFouls", "HomeRedcards", "HomeShots", "HomeShotsOnTarget","HomeYellowcards"]

   df_temp1 = pd.DataFrame()

   for Header in HomeHeaders:   

       df_temp1[Header] = df.groupby(["Season", "HomeTeam"])[Header].sum()

   del HomeHeaders

   AwayHeaders = ["FT_AwayWin", "HT_AwayWin", "FT_AwayDraw",

           "HT_AwayDraw", "FT_AwayLoss", "HT_AwayLoss", "FT_AwayGoals", "HT_AwayGoals", "FT_AwayConceded", "HT_AwayConceded", 

           "AwayCorners", "AwayFouls", "AwayRedcards", "AwayShots", "AwayShotsOnTarget","AwayYellowcards"]

   df_temp2 = pd.DataFrame()

   for Header in AwayHeaders:   

       df_temp2[Header] = df.groupby(["Season", "AwayTeam"])[Header].sum()

   del AwayHeaders

   del Header   

   df_temp = pd.concat([df_temp1, df_temp2], axis=1)

   del df_temp1

   del df_temp2  

   df_temp["HomePoints"] = (3*df_temp["FT_HomeWin"])+(df_temp["FT_HomeDraw"])

   df_temp["AwayPoints"] = (3*df_temp["FT_AwayWin"])+(df_temp["FT_AwayDraw"])

   df_temp["TotalPoints"] = df_temp["HomePoints"] + df_temp["AwayPoints"]

   df_temp["FT_TotalGoals"] =  df_temp["FT_HomeGoals"] + df_temp["FT_AwayGoals"]

   df_temp["FT_TotalConceded"] =  df_temp["FT_HomeConceded"] + df_temp["FT_AwayConceded"]

   df_temp["FT_GoalDifference"] =  df_temp["FT_TotalGoals"] - df_temp["FT_TotalConceded"]

   df_temp["HT_TotalGoals"] =  df_temp["HT_HomeGoals"] + df_temp["HT_AwayGoals"]

   df_temp["HT_TotalConceded"] =  df_temp["HT_HomeConceded"] + df_temp["HT_AwayConceded"]

   df_temp["HT_GoalDifference"] =  df_temp["HT_TotalGoals"] - df_temp["HT_TotalConceded"]

   df_temp.reset_index(level=0, inplace=True)

   df_temp.reset_index(level=0, inplace=True)

   #----------------------------------------------------------------------

   #Create a column that shows which place each team finished in the table each season

   #----------------------------------------------------------------------

   df_temp["Season"] = df_temp["Season"].astype(int)

   df_temp.sort_values(["Season", "TotalPoints", "FT_GoalDifference", "FT_TotalGoals"], ascending=[True, False, False, False], inplace=True)

   seasons = df_temp["Season"].unique().tolist()

   placelist = []

   for season in seasons:

       teams = df_temp[df_temp.Season == season]["HomeTeam"].unique().tolist()

       placement = 0

       lst = []

       for team in teams:

           placement = placement + 1

           lst.append(placement)

       placelist.extend(lst)

   df_temp["Placement"] = placelist

   del lst

   del placelist

   del placement

   del season

   del seasons

   del team

   del teams



   return df_temp

df2 = TableDataFrame(df)

def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.0f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)   

#------------------------------------------------------------------------



df2 = TableDataFrame(df)



#--------------Create Dataframe for top 6 teams--------------------------

Top6 = ['Arsenal','Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']

df_top6 = df[df.HomeTeam.isin(Top6) & df.AwayTeam.isin(Top6)]

#-----------------------------------------------------------------------



#-----------------------------------------------------------------------

#---------------------Plots---------------------------------------------

#---------- Heat Map





def Heatmap(DF, Label ="Heatmap", cmap = sns.diverging_palette(250, 10, n=9, as_cmap=True)):

    corr = DF.corr()

    

    mask = mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    

    plt.figure()

    plt.rcParams["figure.figsize"]=(25,16)

    HeatMap = sns.heatmap(corr, annot = True, linewidths=.6, cmap=cmap, mask=mask)

    HeatMap.set_title(label=Label, fontsize=20)

    

    return HeatMap;
plt.figure()

plt.rcParams["figure.figsize"] = (15,10)

Heatmap(df[['FT_AwayGoals', 'FT_HomeGoals', 'FT_Result', 'HT_AwayGoals', 'HT_HomeGoals',

       'HT_Result', 'HomeCorners', 'HomeFouls', 'HomeRedcards', 'HomeShots',

       'HomeShotsOnTarget', 'HomeYellowcards', 'AwayCorners', 'AwayFouls',

       'AwayRedcards', 'AwayShots', 'AwayShotsOnTarget', 'AwayYellowcards',]])

sns.pairplot(df[~df.Season.isin(NoDeepStats)], x_vars=["FT_HomeGoals", "FT_AwayGoals", "HomeFouls", "HomeRedcards", "HomeYellowcards", "AwayFouls", "AwayRedcards", "AwayYellowcards"]

                    , y_vars=["FT_HomeGoals", "FT_AwayGoals", "HomeFouls", "HomeRedcards", "HomeYellowcards", "AwayFouls", "AwayRedcards", "AwayYellowcards"]

                    , kind="reg" ,diag_kind = 'hist' );
sns.pairplot(df[~df.Season.isin(NoDeepStats)], x_vars=["FT_HomeGoals", "FT_AwayGoals", "HomeCorners", "HomeShots", "HomeShotsOnTarget", "AwayCorners", "AwayShots", "AwayShotsOnTarget"]

                    , y_vars=["FT_HomeGoals", "FT_AwayGoals", "HomeCorners", "HomeShots", "HomeShotsOnTarget", "AwayCorners", "AwayShots", "AwayShotsOnTarget"]

                    , kind="reg",diag_kind = 'hist'); 
plt.figure(figsize=(10,10))

df.FT_Result.value_counts().plot.pie(autopct='%1.1f%%',shadow=True,cmap='Pastel1')

plt.figure(figsize=(25,10))

df3 = pd.DataFrame(df2.groupby('HomeTeam')['TotalPoints'].sum()).reset_index()

df3.sort_values(["TotalPoints"], ascending=[False], inplace=True)

g = sns.barplot(x="HomeTeam", y="TotalPoints", data=df3)

plt.xticks(rotation=90)

show_values_on_bars(g)

plt.show()
df3 = pd.DataFrame(df2.groupby('HomeTeam')['TotalPoints', "FT_HomeWin", "FT_HomeDraw", "FT_HomeLoss", 

                   "FT_AwayWin", "FT_AwayDraw", "FT_AwayLoss", "FT_GoalDifference", 'FT_TotalGoals'].sum()).reset_index()

df3["Win"] = df3["FT_HomeWin"] + df3["FT_AwayWin"]

df3["Draw"] = df3["FT_HomeDraw"] + df3["FT_AwayDraw"]

df3["Loss"] = df3["FT_HomeLoss"] + df3["FT_AwayLoss"]

df3["NoGames"] = df3["Win"] + df3["Draw"] + df3["Loss"]

df3.sort_values(["TotalPoints", "FT_GoalDifference"], ascending=[False, False], inplace=True)



print(df3[["HomeTeam", "TotalPoints", "NoGames", "Win", "Draw", "Loss","FT_GoalDifference"]])
g = sns.catplot("HomeTeam", data=df2, kind="count", order = df2['HomeTeam'].value_counts().index)

g.set_xticklabels(rotation=270)

g.fig.set_size_inches(25,20)
df2['Season'] = df2['Season'].astype(str)

plt.figure(figsize=(25,10))

plt.plot( 'Season', 'Placement', data=df2.query("HomeTeam == 'Chelsea'"), marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label="Chelsea")

plt.plot( 'Season', 'Placement', data=df2.query("HomeTeam == 'Arsenal'"), marker='o', markerfacecolor='white', markersize=12, color='red', linewidth=4, label="Arsenal")

plt.plot( 'Season', 'Placement', data=df2.query("HomeTeam == 'Man City'"), marker='o', markerfacecolor='white', markersize=12, color='teal', linewidth=4, label="Man City")

plt.plot( 'Season', 'Placement', data=df2.query("HomeTeam == 'Tottenham'"), marker='o', markerfacecolor='white', markersize=12, color='grey', linewidth=4, label="Tottenham")

plt.plot( 'Season', 'Placement', data=df2.query("HomeTeam == 'Man United'"), marker='o', markerfacecolor='black', markersize=12, color='red', linewidth=4, label="Man United")

plt.plot( 'Season', 'Placement', data=df2.query("HomeTeam == 'Liverpool'"), marker='o', markerfacecolor='red', markersize=12, color='red', linewidth=4, label="Liverpool")

plt.xticks(rotation=45)

plt.gca().invert_yaxis()

plt.yticks([20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1])

plt.legend()
HomeHeaders = ["FT_HomeWin", "HT_HomeWin", "FT_HomeDraw", "HT_HomeDraw",

           "FT_HomeLoss", "HT_HomeLoss", "FT_HomeGoals", "HT_HomeGoals", "FT_HomeConceded", "HT_HomeConceded", 

              "HomeCorners", "HomeFouls", "HomeRedcards", "HomeShots", "HomeShotsOnTarget","HomeYellowcards"]

df_temp1 = pd.DataFrame()

for Header in HomeHeaders:   

    df_temp1[Header] = df[~df.Season.isin(NoDeepStats)].groupby(["Season"])[Header].sum()

del HomeHeaders

AwayHeaders = ["FT_AwayWin", "HT_AwayWin", "FT_AwayDraw",

           "HT_AwayDraw", "FT_AwayLoss", "HT_AwayLoss", "FT_AwayGoals", "HT_AwayGoals", "FT_AwayConceded", "HT_AwayConceded", 

           "AwayCorners", "AwayFouls", "AwayRedcards", "AwayShots", "AwayShotsOnTarget","AwayYellowcards"]

df_temp2 = pd.DataFrame()

for Header in AwayHeaders:   

    df_temp2[Header] = df[~df.Season.isin(NoDeepStats)].groupby(["Season"])[Header].sum()

del AwayHeaders

del Header   

df3 = pd.concat([df_temp1, df_temp2], axis=1)

del df_temp1

del df_temp2 

df3.reset_index(level=0, inplace=True)



df3["TotalYellowcards"] = df3["HomeYellowcards"] + df3["AwayYellowcards"]

df3["TotalRedcards"] = df3["HomeRedcards"] + df3["AwayRedcards"]



plt.figure(figsize=(25,10))

plt.plot( 'Season', 'TotalYellowcards', data=df3, marker='o', markerfacecolor='yellow', markersize=12, color='yellow', linewidth=4, label="Total Yellow Cards")

plt.plot( 'Season', 'HomeYellowcards', data=df3, marker='o', markerfacecolor='black', markersize=12, color='yellow', linewidth=4, label="Home Yellow Cards")

plt.plot( 'Season', 'AwayYellowcards', data=df3, marker='o', markerfacecolor='white', markersize=12, color='yellow', linewidth=4, label="Away Yellow Cards")

plt.plot( 'Season', 'TotalRedcards', data=df3, marker='o', markerfacecolor='red', markersize=12, color='red', linewidth=4, label="Total Red Cards")

plt.plot( 'Season', 'HomeRedcards', data=df3, marker='o', markerfacecolor='black', markersize=12, color='red', linewidth=4, label="Home Red Cards")

plt.plot( 'Season', 'AwayRedcards', data=df3, marker='o', markerfacecolor='white', markersize=12, color='red', linewidth=4, label="Away Red Cards")



plt.xticks(rotation=45)

plt.legend(loc='best')
df3["TotalFouls"] = df3["HomeFouls"] + df3["AwayFouls"]
plt.figure(figsize=(25,10))

plt.plot( 'Season', 'TotalFouls', data=df3, marker='o', markerfacecolor='black', markersize=12, color='green', linewidth=4, label="Total Fouls")

plt.plot( 'Season', 'HomeFouls', data=df3, marker='o', markerfacecolor='black', markersize=12, color='blue', linewidth=4, label="Home Fouls")

plt.plot( 'Season', 'AwayFouls', data=df3, marker='o', markerfacecolor='black', markersize=12, color='red', linewidth=4, label="Away Fouls")



plt.xticks(rotation=45)

plt.legend(loc='best')
plt.figure()

plt.rcParams["figure.figsize"] = (15,10)

Heatmap(df3[['TotalFouls', 'TotalYellowcards', 'TotalRedcards', 'HomeFouls', 'HomeRedcards','HomeYellowcards','AwayFouls',

       'AwayRedcards','AwayYellowcards',]])

plt.figure(figsize=(25,10))

sns.countplot(x="Season", hue="FT_Result", data=df)


plt.figure(figsize=(25,10))

plt.plot( 'Season', 'TotalPoints', data=df2[(df2.Season != "19931994") & (df2.Season != "19941995") & (df2.Placement == 1)], marker='o', markerfacecolor='green', markersize=12, color='green', linewidth=4, label="Points to win the league")

plt.xticks(rotation=45)



plt.figure(figsize=(25,10))

plt.plot( 'Season', 'TotalPoints', data=df2[(df2.Season != "19931994") & (df2.Season != "19941995") & (df2.Placement == 17)], marker='o', markerfacecolor='green', markersize=12, color='green', linewidth=4, label="Finished above relegation")

plt.plot( 'Season', 'TotalPoints', data=df2[(df2.Season != "19931994") & (df2.Season != "19941995") & (df2.Placement > 17)], marker='o', markerfacecolor='red', markersize=12, color='red', linewidth=4, label="Relegated")

plt.xticks(rotation=45)

plt.legend()

df3["TotalShotsOnTarget"] = df3["HomeShotsOnTarget"] + df3["AwayShotsOnTarget"]

df3["TotalShots"] = df3["HomeShots"] + df3["AwayShots"]



plt.figure(figsize=(25,10))

plt.plot( 'Season', 'TotalShotsOnTarget', data=df3, marker='o', markerfacecolor='green', markersize=12, color='green', linewidth=4, label="Total Shots on target")

plt.plot( 'Season', 'HomeShotsOnTarget', data=df3, marker='o', markerfacecolor='black', markersize=12, color='green', linewidth=4, label="Home Shots on target")

plt.plot( 'Season', 'AwayShotsOnTarget', data=df3, marker='o', markerfacecolor='white', markersize=12, color='green', linewidth=4, label="Away Shots on target")

plt.plot( 'Season', 'TotalShots', data=df3, marker='o', markerfacecolor='blue', markersize=12, color='blue', linewidth=4, label="Total Shots")

plt.plot( 'Season', 'HomeShots', data=df3, marker='o', markerfacecolor='black', markersize=12, color='blue', linewidth=4, label="Home Shots")

plt.plot( 'Season', 'AwayShots', data=df3, marker='o', markerfacecolor='white', markersize=12, color='blue', linewidth=4, label="Away Shots")

df3["TotalGoals"] = df3["FT_HomeGoals"] + df3["FT_AwayGoals"]

plt.plot( 'Season', 'TotalGoals', data=df3, marker='o', markerfacecolor='green', markersize=12, color='purple', linewidth=4, label="Total Goals")

plt.plot( 'Season', 'FT_HomeGoals', data=df3, marker='o', markerfacecolor='black', markersize=12, color='purple', linewidth=4, label="Home Goals")

plt.plot( 'Season', 'FT_AwayGoals', data=df3, marker='o', markerfacecolor='white', markersize=12, color='purple', linewidth=4, label="Away Goals")

plt.xticks(rotation=45)

plt.legend()
#Creating some new dataframes to use for analysis

#-----Create new dataframe for full seasons--------------

# It's probably overkill to create this as a function since I'm not having the groupby-columns as input to the function. But I might change this at a latter stage.



def TableDataFrame2(df): 

   HomeHeaders = ["FT_HomeWin", "HT_HomeWin", "FT_HomeDraw", "HT_HomeDraw",

           "FT_HomeLoss", "HT_HomeLoss", "FT_HomeGoals", "HT_HomeGoals", "FT_HomeConceded", "HT_HomeConceded", 

              "HomeCorners", "HomeFouls", "HomeRedcards", "HomeShots", "HomeShotsOnTarget","HomeYellowcards"]

   df_temp1 = pd.DataFrame()

   for Header in HomeHeaders:   

       df_temp1[Header] = df.groupby(["Season", "HomeTeam", "Referee"])[Header].sum()

   del HomeHeaders

   AwayHeaders = ["FT_AwayWin", "HT_AwayWin", "FT_AwayDraw",

           "HT_AwayDraw", "FT_AwayLoss", "HT_AwayLoss", "FT_AwayGoals", "HT_AwayGoals", "FT_AwayConceded", "HT_AwayConceded", 

           "AwayCorners", "AwayFouls", "AwayRedcards", "AwayShots", "AwayShotsOnTarget","AwayYellowcards"]

   df_temp2 = pd.DataFrame()

   for Header in AwayHeaders:   

       df_temp2[Header] = df.groupby(["Season", "AwayTeam", "Referee"])[Header].sum()

   del AwayHeaders

   del Header   

   df_temp = pd.concat([df_temp1, df_temp2], axis=1)

   del df_temp1

   del df_temp2  

   df_temp["HomePoints"] = (3*df_temp["FT_HomeWin"])+(df_temp["FT_HomeDraw"])

   df_temp["AwayPoints"] = (3*df_temp["FT_AwayWin"])+(df_temp["FT_AwayDraw"])

   df_temp["TotalPoints"] = df_temp["HomePoints"] + df_temp["AwayPoints"]

   df_temp["FT_TotalGoals"] =  df_temp["FT_HomeGoals"] + df_temp["FT_AwayGoals"]

   df_temp["FT_TotalConceded"] =  df_temp["FT_HomeConceded"] + df_temp["FT_AwayConceded"]

   df_temp["FT_GoalDifference"] =  df_temp["FT_TotalGoals"] - df_temp["FT_TotalConceded"]

   df_temp["HT_TotalGoals"] =  df_temp["HT_HomeGoals"] + df_temp["HT_AwayGoals"]

   df_temp["HT_TotalConceded"] =  df_temp["HT_HomeConceded"] + df_temp["HT_AwayConceded"]

   df_temp["HT_GoalDifference"] =  df_temp["HT_TotalGoals"] - df_temp["HT_TotalConceded"]

   df_temp.reset_index(level=2, inplace=True)

   df_temp.reset_index(level=1, inplace=True)

   df_temp.reset_index(level=0, inplace=True)

   df_temp.rename(columns={'index':'Season',

                          'level_1':'Team',

                          'level_2':'Referee'}, 

                 inplace=True)



   return df_temp



#----------------------------------------------------------------------



#-----Create new dataframe for full dataset-------------



def TableDataFrame3(df): 

   HomeHeaders = ["FT_HomeWin", "HT_HomeWin", "FT_HomeDraw", "HT_HomeDraw",

           "FT_HomeLoss", "HT_HomeLoss", "FT_HomeGoals", "HT_HomeGoals", "FT_HomeConceded", "HT_HomeConceded", 

              "HomeCorners", "HomeFouls", "HomeRedcards", "HomeShots", "HomeShotsOnTarget","HomeYellowcards"]

   df_temp1 = pd.DataFrame()

   for Header in HomeHeaders:   

       df_temp1[Header] = df.groupby(["HomeTeam", "Referee"])[Header].sum()

   del HomeHeaders

   AwayHeaders = ["FT_AwayWin", "HT_AwayWin", "FT_AwayDraw",

           "HT_AwayDraw", "FT_AwayLoss", "HT_AwayLoss", "FT_AwayGoals", "HT_AwayGoals", "FT_AwayConceded", "HT_AwayConceded", 

           "AwayCorners", "AwayFouls", "AwayRedcards", "AwayShots", "AwayShotsOnTarget","AwayYellowcards"]

   df_temp2 = pd.DataFrame()

   for Header in AwayHeaders:   

       df_temp2[Header] = df.groupby(["AwayTeam", "Referee"])[Header].sum()

   del AwayHeaders

   del Header   

   df_temp = pd.concat([df_temp1, df_temp2], axis=1)

   del df_temp1

   del df_temp2  

   df_temp["HomeRatio"] = (df_temp["FT_HomeWin"])/(df_temp["FT_HomeWin"] + df_temp["FT_HomeDraw"] + df_temp["FT_HomeLoss"])

   df_temp["AwayRatio"] = (df_temp["FT_AwayWin"])/(df_temp["FT_AwayWin"] + df_temp["FT_AwayDraw"] + df_temp["FT_AwayLoss"])

   df_temp["TotalRatio"] = (df_temp["FT_HomeWin"] + df_temp["FT_AwayWin"]) / (df_temp["FT_HomeWin"] + df_temp["FT_AwayWin"] + df_temp["FT_HomeDraw"] + df_temp["FT_HomeLoss"] + df_temp["FT_AwayDraw"] + df_temp["FT_AwayLoss"])

   df_temp["NoGames"] =  (df_temp["FT_HomeWin"] + df_temp["FT_AwayWin"] + df_temp["FT_HomeDraw"] + df_temp["FT_HomeLoss"] + df_temp["FT_AwayDraw"] + df_temp["FT_AwayLoss"])

   

   df_temp["AwayYCardperFoul"] = (df_temp["AwayYellowcards"])/(df_temp["AwayFouls"])

   df_temp["AwayRCardperFoul"] = (df_temp["AwayRedcards"])/(df_temp["AwayFouls"])

   df_temp["HomeYCardperFoul"] = (df_temp["HomeYellowcards"])/(df_temp["HomeFouls"])

   df_temp["HomeRCardperFoul"] = (df_temp["HomeRedcards"])/(df_temp["HomeFouls"])

   df_temp["TotalYCardperFoul"] = (df_temp["AwayYellowcards"]+df_temp["HomeYellowcards"])/(df_temp["AwayFouls"]+df_temp["HomeFouls"])

   df_temp["TotalRCardperFoul"] = (df_temp["AwayRedcards"]+df_temp["HomeRedcards"])/(df_temp["AwayFouls"]+df_temp["HomeFouls"])    

   df_temp.reset_index(level=1, inplace=True)

   df_temp.reset_index(level=0, inplace=True)

   df_temp.rename(columns={'index':'Team',

                          'level_1':'Referee'}, 

                 inplace=True)



   return df_temp





df2 = TableDataFrame2(df)

df3 = TableDataFrame3(df)


# Function to show values on barplot

#----------------------------------------------------------------------



def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.0f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)   

#------------------------------------------------------------------------
Header = ["FT_HomeGoals", "FT_HomeConceded", "HomeFouls", "HomeRedcards", "HomeYellowcards", "FT_HomeWin", "FT_HomeDraw", "FT_HomeLoss", "FT_AwayGoals", "FT_AwayConceded", "AwayFouls", "AwayRedcards", "AwayYellowcards", "FT_AwayWin", "FT_AwayDraw", "FT_AwayLoss"]

df4 = pd.DataFrame()

df4["NoGames"] = df["Referee"].value_counts()

for Header in Header:

    df4[Header] = df.groupby(["Referee"])[Header].sum()

df4.reset_index(level=0, inplace=True)

df4["WinRatio"] = df4["FT_HomeWin"] / df4["FT_HomeLoss"]

df4["GoalRatio"] = (df4["FT_HomeGoals"] + df4["FT_HomeConceded"]) / df4["NoGames"]

plt.figure()

plt.rcParams["figure.figsize"] = (15,10)

plt.xticks(rotation=90)

sns.barplot(x="index", y="WinRatio", data=df4[df4.NoGames > 99])
plt.figure()

ax = sns.barplot(x="Team", y="TotalRatio", data=df3[df3.NoGames > 10][df3.Referee == "Jones"])

plt.xticks(rotation=90)

type(ax)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])



print(df.query("Referee == 'Jones'").query("HomeTeam == 'Arsenal' | AwayTeam == 'Arsenal'")[["Season", "HomeTeam", "AwayTeam","FT_Result","IWH", "IWD", "IWA"]])
plt.figure()

ax = sns.barplot(x="Team", y="TotalRatio", data=df3[df3.NoGames > 10][df3.Referee == "Swarbrick"])

plt.xticks(rotation=90)

type(ax)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
print(df.query("Referee == 'Swarbrick'").query("HomeTeam == 'Aston Villa' | AwayTeam == 'Aston Villa'")[["Season", "HomeTeam", "AwayTeam","FT_Result","IWH", "IWD", "IWA"]])
plt.figure()

plt.rcParams["figure.figsize"] = (15,10)

plt.xticks(rotation=90)

sns.barplot(x="index", y="GoalRatio", data=df4[df4.NoGames > 99])

plt.figure()

ax = sns.barplot(x="Referee", y="TotalRatio", data=df3[df3.NoGames > 10])

plt.xticks(rotation=90)

type(ax)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)

ax1.title.set_text('Barber')

ax2.title.set_text('Dunn')

ax3.title.set_text('Attwell')

ax4.title.set_text('Dean')



ax = sns.barplot(x="Team", y="TotalRatio", data=df3[df3.NoGames > 10][df3.Referee == "Barber"], ax=ax1)

type(ax)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

ax = sns.barplot(x="Team", y="TotalRatio", data=df3[df3.NoGames > 10][df3.Referee == "Dunn"], ax=ax2)

type(ax)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

ax = sns.barplot(x="Team", y="TotalRatio", data=df3[df3.NoGames > 10][df3.Referee == "Attwell"], ax=ax3)

type(ax)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

ax = sns.barplot(x="Team", y="TotalRatio", data=df3[df3.NoGames > 10][df3.Referee == "Dean"], ax=ax4)

type(ax)

plt.setp(ax4.get_xticklabels(), rotation=90)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
#-------------------------------------------------------

plt.figure()

ax = sns.barplot(x="Referee", y="TotalYCardperFoul", data=df3[df3.NoGames > 10])

plt.xticks(rotation=90)

type(ax)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.figure()

ax = sns.barplot(x="Referee", y="TotalRCardperFoul", data=df3[df3.NoGames > 10])

plt.xticks(rotation=90)

type(ax)

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])