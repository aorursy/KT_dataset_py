#import modules

import pandas as pd

import numpy as np



#load dataframe

df_2 = pd.read_csv("../input/march-madness-mascots/AllMascots.csv", low_memory = False)



#drop unnecessary id column

df_2.drop('rowid', axis=1)
# import module

import plotly.express as px



# set plot parameters

fig = px.histogram(df_2, y="MascotType", histnorm = 'percent', orientation = 'h', height = 2000)

fig.data[0].marker.color = "orange"

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



# set plot title

fig.update_layout(

    title={

        'text': "U.S. college mascot types divided by percentage",

        'y':0.99,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



# show plot

fig.show()
# print value count of MascotType column

df_2['MascotType'].value_counts()
# load dataframe

df = pd.read_csv("../input/march-madness-mascots/MarchMascots.csv", low_memory = False)



# drop unnecessary id column

df = df.drop('rowid', axis=1)



# printout

df.head(10)
# set plot parameters

fig = px.histogram(df, y="MascotType", histnorm = 'percent', orientation = 'h', height = 2000)

fig.data[0].marker.color = "orange"

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



# set plot title

fig.update_layout(

    title={

        'text': "March Madness mascots 2015-2019 divided by mascot type",

        'y':0.99,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



# show plot

fig.show()
# print value count of MascotType column

df['MascotType'].value_counts()
# load dataframe

df_mseed = pd.read_csv('../input/ncaa-data/MNCAATourneySeeds.csv', low_memory = False)



# select rows with season value 2015 or larger (dataframe ends in 2019)

df_m1519 = df_mseed.loc[df_mseed['Season'] >= 2015]



# rename TeamID column to fit the mascots MTeamID column

df_m1519b = df_m1519.rename(columns = {'TeamID':'MTeamID'}) 



# printout

df_m1519b.head(10)
# load dataframe

df_wseed = pd.read_csv('../input/ncaa-data/WNCAATourneySeeds.csv', low_memory = False)



# select rows with season value 2015 or larger (dataframe ends in 2019)

df_w1519 = df_wseed.loc[df_wseed['Season'] >= 2015]



# rename TeamID column to fit the mascots MTeamID column

df_w1519b = df_w1519.rename(columns = {'TeamID':'WTeamID'}) 



# printout

df_w1519b.head(10)
# merge two dataframes on common rows

# in SQL the same would be done by using left join

df_Mseeded = pd.merge(df_m1519b, df, on='MTeamID', how='left')



# drop unnecessary columns

unused_cols = ['WTeamID']

df_Mseeded = df_Mseeded.drop(unused_cols, axis=1)



# printout

df_Mseeded.head(10)
# set plot parameters

fig = px.histogram(df_Mseeded, y="MascotType", histnorm = 'percent', orientation = 'h', height = 2000)

fig.data[0].marker.color = "orange"

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



# set plot title

fig.update_layout(

    title={

        'text': "Men's March Madness mascots 2015-2019 divided by mascot type",

        'y':0.99,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



# show plot

fig.show()
#merge two dataframes on common rows

df_Wseeded = pd.merge(df_w1519b, df, on='WTeamID', how='left')



# drop unnecessary columns

unused_cols2 = ['MTeamID']

df_Wseeded = df_Wseeded.drop(unused_cols2, axis=1)



# set plot parameters

fig = px.histogram(df_Wseeded, y="MascotType", histnorm = 'percent', orientation = 'h', height = 2000)

fig.data[0].marker.color = "orange"

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



# set plot title etc.

fig.update_layout(

    title={

        'text': "Women's March Madness mascots 2015-2019 divided by mascot type",

        'y':0.99,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



# show plot

fig.show()
# select all rows with value 'Villanova' in TeamName column

df_Wseeded.loc[df_Wseeded['TeamName'] == 'Villanova']
# load dataframe

df_Wresults = pd.read_csv('../input/ncaa-data/WNCAATourneyCompactResults.csv', low_memory = False)



# select rows with season value 2018

df_Wresults2018 = df_Wresults.loc[df_Wresults['Season'] == 2018]



# select all rows with Villanova women's team ID (3437) in the LTeamID column

df_Wresults2018.loc[df_Wresults2018['LTeamID'] == 3437]
# from MarchMascots dataframe select row or rows with women's team ID 3323

df.loc[df['WTeamID'] == 3323]
# select rows with season value 2015 up to latest year in dataset (2019)

df_Wresults2015_ = df_Wresults.loc[df_Wresults['Season'] >= 2015]



# select all rows where either WTeamID or LTeamID is Notre Dame women's team ID (3323) 

df_Wresults2015_ =  df_Wresults2015_[np.logical_or(df_Wresults2015_['WTeamID'] == 3323, df_Wresults2015_['LTeamID'] == 3323)]



# show results

df_Wresults2015_
# import modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# set plot size etc.

sns.set(rc={'figure.figsize':(9.7,8.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(x = 'Season',

              data = df_Wresults2015_,

              order = df_Wresults2015_['Season'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Notre Dame women's team in March Madness 2015-2019",fontsize=20)

plot.set_xlabel("Year",fontsize=18)

plot.set_ylabel("Number of games played per year",fontsize=18)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# select all rows where either WTeamID or LTeamID is Notre Dame women's team ID (3323) 

df_Wresults =  df_Wresults[np.logical_or(df_Wresults['WTeamID'] == 3323, df_Wresults['LTeamID'] == 3323)]



# set plot size etc.

sns.set(rc={'figure.figsize':(16.7,8.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(x = 'Season',

              data = df_Wresults,

              order = df_Wresults['Season'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Notre Dame women's team in March Madness 1998-2019",fontsize=20)

plot.set_xlabel("Year",fontsize=18)

plot.set_ylabel("Number of games played per year",fontsize=18)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# load data

df_Wfinalists = pd.read_csv('../input/ncaa-data/WNCAATourneyCompactResults.csv', low_memory = False)



#select rows with season value 2015 up to latest year in dataset (2019)

df_Wfinalists2015_ = df_Wfinalists.loc[df_Wfinalists['Season'] >= 2015]



# list teams with most frequently show up in WTeamID column

df_Wfinalists2015_['WTeamID'].value_counts()




# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,20.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'WTeamID',

              data = df_Wfinalists2015_,

              order = df_Wfinalists2015_['WTeamID'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Number of March Madness games played by women's teams 2015-2019",fontsize=20)

plot.set_xlabel("Number of games",fontsize=18)

plot.set_ylabel("Team ID",fontsize=16)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# merge dataframes

df_BestWomen = pd.merge(df_Wfinalists2015_, df, on='WTeamID', how='left')



# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,22.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'MascotName',

              data = df_BestWomen,

              order = df_BestWomen['MascotName'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Mascots with most appearances in women's March Madness tournament 2015-2019",fontsize=20)

plot.set_xlabel("Number of appearances",fontsize=18)

plot.set_ylabel("Mascot name",fontsize=16)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# get only the first count of a row value in a column

df_Wuniques = df.drop_duplicates(subset='College', keep="first")



# merge dataframes

df_BestWomen = pd.merge(df_Wfinalists2015_, df_Wuniques, on='WTeamID', how='left')



# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,22.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'College',

              data = df_BestWomen,

              order = df_BestWomen['College'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Games per college in women's March Madness tournament 2015-2019",fontsize=20)

plot.set_xlabel("Number of games",fontsize=18)

plot.set_ylabel("College",fontsize=16)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# load dataframe

df_Mresults = pd.read_csv('../input/ncaa-data/MNCAATourneyCompactResults.csv', low_memory = False)



# select rows with season value 2015 or larger

df_m1519 = df_mseed.loc[df_mseed['Season'] >= 2015]



# select rows with season value 2015 up to latest year

df_Mresults2015_ = df_Mresults.loc[df_Mresults['Season'] >= 2015]



# list teams with most frequently show up in WTeamID column

df_Mresults2015_['WTeamID'].value_counts()
# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,26.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'WTeamID',

              data = df_Mresults2015_,

              order = df_Mresults2015_['WTeamID'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Number of March Madness games played by men's teams 2015-2019",fontsize=20)

plot.set_xlabel("Number of games",fontsize=18)

plot.set_ylabel("Team ID",fontsize=16)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# rename TeamID column to fit the mascots dataframe MTeamID column

df_Mwinners = df_Mresults2015_.rename(columns = {'WTeamID':'MTeamID'}) 
# merge dataframes

df_BestMen = pd.merge(df_Mwinners, df, on='MTeamID', how='left')



# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,32.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'MascotName',

              data = df_BestMen,

              order = df_BestMen['MascotName'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Mascots with most appearances in men's March Madness tournament 2015-2019",fontsize=20)

plot.set_xlabel("Number of appearances",fontsize=18)

plot.set_ylabel("Mascot name",fontsize=16)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# get only the first count of a row value in a column

df_Muniques = df.drop_duplicates(subset='College', keep='first')



# merge dataframes

df_BestMen_ = pd.merge(df_Mwinners, df_Muniques, on='MTeamID', how='left')



# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,22.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'College',

              data = df_BestMen_,

              order = df_BestMen_['College'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Games per college in men's March Madness tournament 2015-2019",fontsize=20)

plot.set_xlabel("Number of games",fontsize=18)

plot.set_ylabel("College",fontsize=16)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,20.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'MascotType',

              data = df_BestMen_,

              order = df_BestMen_['MascotType'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Games per mascot type in men's March Madness tournament 2015-2019",fontsize=20)

plot.set_xlabel("Number of games",fontsize=18)

plot.set_ylabel("Mascot type",fontsize=16)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# count the values in MascotType column by percentage

(df_BestMen_['MascotType'].value_counts()/df_BestMen_['MascotType'].count())*100
# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,20.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'MascotType',

              data = df_BestWomen,

              order = df_BestWomen['MascotType'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Games per mascot type in women's March Madness tournament 2015-2019",fontsize=20)

plot.set_xlabel("Number of games",fontsize=18)

plot.set_ylabel("Mascot type",fontsize=16)

plot.tick_params(labelsize=14)



# show plot

plt.show()
# count the values in MascotType column by percentage

(df_BestWomen['MascotType'].value_counts()/df_BestWomen['MascotType'].count())*100
# printout

print (df.MascotType.unique())
# load dataframe

df_classes = pd.read_csv("../input/march-madness-mascots/MarchMascots.csv", low_memory = False)



# drop unnecessary id columna (not necessary but clarifies the dataset)

idcolumns = ['rowid']

df_classes = df_classes.drop(idcolumns, axis=1)



# create a new column with value 0 or 1 depending on whether mascot is a bird

df_classes['ClassBird'] = df_classes.MascotType.apply(lambda x: 1 if x in [

    'eagle',

    'cardinal',

    'falcon',

    'bird',

    'rooster',

    'burrowing owl',

    'owl',

    'bald eagle',

    'red bird',

    'mountain hawk',

    'hawk',

    'thunderbird',

    'pelican',

    'mockingbird',

    'blue hen',

    'osprey',

    'duck',

    'commodore',

    'emperor penguin'

] else 0)
# printout

df_classes.head(10)
# count the values in ClassBird column by percentage

(df_classes['ClassBird'].value_counts()/df_classes['ClassBird'].count())*100
# create a new column with value 0 or 1 depending on whether mascot is a human

df_classes['ClassHuman'] = df_classes.MascotType.apply(lambda x: 2 if x in [

    'human',

] else 0)



# count the values in ClassHuman column by percentage

(df_classes['ClassHuman'].value_counts()/df_classes['ClassHuman'].count())*100
# create a new column with value 0 or 1 depending on whether mascot is a wildcat

df_classes['ClassWildcat'] = df_classes.MascotType.apply(lambda x: 3 if x in [

    'wildcat',

    'tiger',

    'cougar',

    'lion',

    'jaguar',

    'bobcat',

    'tiger mascot'  

] else 0)



# count the values in ClassWildcat column by percentage

(df_classes['ClassWildcat'].value_counts()/df_classes['ClassWildcat'].count())*100
# create a new column with value 0 or 1 depending on whether mascot is a bird

df_classes['ClassDog'] = df_classes.MascotType.apply(lambda x: 4 if x in [

    'dog',

    'terrier',

    'bulldog',

    'husky',

    'greyhound',

    'Saint Bernard dog',

    'English bulldog',

    'hound'

] else 0)



# count the values in ClassHuman column by percentage

(df_classes['ClassDog'].value_counts()/df_classes['ClassDog'].count())*100
# create column AllClasses from mascot class columns

df_classes["AllClasses"] = (df_classes["ClassBird"] + df_classes["ClassHuman"] + df_classes["ClassDog"] + df_classes["ClassWildcat"]).astype("int")



# print first rows

df_classes.head(10)
# count the values in AllClasses column by percentage

(df_classes['AllClasses'].value_counts()/df_classes['AllClasses'].count())*100
# merge dataframes

df_ClassWomen = pd.merge(df_Wfinalists2015_, df_classes, on='WTeamID', how='left')



# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,7.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'AllClasses',

              data = df_ClassWomen,

              order = df_ClassWomen['AllClasses'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Games per mascot type in women's March Madness tournament 2015-2019",fontsize=20)

plot.set_xlabel("Number of games",fontsize=18)

plot.set_ylabel("Mascot type",fontsize=16)

plot.tick_params(labelsize=14)

plot.legend (loc=4, fontsize = 16, fancybox=True, framealpha=1, shadow=True, borderpad=1, 

             title = '0=other, 1=bird, 2=human, 3=wildcat, 4=dog')



# show plot

plt.show()
# count the values in AllClasses column by percentage

(df_ClassWomen['AllClasses'].value_counts()/df_ClassWomen['AllClasses'].count())*100
# merge dataframes

df_ClassMen = pd.merge(df_Mwinners, df_classes, on='MTeamID', how='left')



# set plot size etc.

sns.set(rc={'figure.figsize':(14.7,9.27)})

sns.set(font='sans-serif', palette='colorblind')



# set plot parameters

plot = sns.countplot(y = 'AllClasses',

              data = df_ClassMen,

              order = df_ClassMen['AllClasses'].value_counts().index)



# set plot title etc.

plot.axes.set_title("Games per mascot type in men's March Madness tournament 2015-2019",fontsize=20)

plot.set_xlabel("Number of games",fontsize=18)

plot.set_ylabel("Mascot type",fontsize=16)

plot.tick_params(labelsize=14)

plot.legend (loc=4, fontsize = 16, fancybox=True, framealpha=1, shadow=True, borderpad=1, 

             title = '0=other, 1=bird, 2=human, 3=wildcat, 4=dog')



# show plot

plt.show()
# count the values in AllClasses column by percentage

(df_ClassMen['AllClasses'].value_counts()/df_ClassMen['AllClasses'].count())*100
# printout

df_m1519b.head(10)
# remove letters from Seeds column

df_m1519c = df_m1519b.replace(regex=['W','X','Y','Z','a','b'], value=' ')



# make Seeds column numeric

df_m1519c['Seed'] = pd.to_numeric(df_m1519c['Seed'])



# check column datatypes

df_m1519c.dtypes
# rename MTeamID column

df_m1519c = df_m1519c.rename(columns = {'MTeamID':'TeamID'}) 



# printout

df_m1519c.head()
# load men's teams dataframe

df_MTeams = pd.read_csv('../input/ncaa-data/MTeams.csv', low_memory = False)



# drop unnecessary columna

dropcolumns = ['FirstD1Season', 'LastD1Season']

df_MTeams = df_MTeams.drop(dropcolumns, axis=1)



# printout

df_MTeams.head()
# merge dataframes

df_SeedNames = pd.merge(df_m1519c, df_MTeams, on='TeamID', how='left')



# printout

df_SeedNames.head()
# select only seeds 13-16

df_LowMen = df_SeedNames[(df_SeedNames['Seed']>=13)]



# select rows with season value 2019

df_LowMen19 = df_LowMen.loc[df_LowMen['Season'] == 2019]



# printout

df_LowMen19.head(8)
# load dataframe

df_Mfinalists = pd.read_csv('../input/ncaa-data/MNCAATourneyCompactResults.csv', low_memory = False)



# select rows with season value 2019

df_Mfinalists2019 = df_Mfinalists.loc[df_Mfinalists['Season'] == 2019]



# drop unnecessary columna

dropcolumns = ['WTeamID','WLoc', 'NumOT']

df_Mfinalists2019 = df_Mfinalists2019.drop(dropcolumns, axis=1)



# rename LTeamID column

df_Mfinalists2019 = df_Mfinalists2019.rename(columns = {'LTeamID':'TeamID'}) 



# printout

df_Mfinalists2019.head(10)
# merge the dataframes

df_LowLosses = pd.merge(df_LowMen19, df_Mfinalists2019, on='TeamID', how='left')



# drop unnecessary columna

dropcolumns = ['Season_y', 'DayNum']

df_LowLosses = df_LowLosses.drop(dropcolumns, axis=1)



# rename Season_x column

df_LowLosses =  df_LowLosses.rename(columns = {'Season_x':'Season'}) 



# drop columns with no value

df_LowLosses = df_LowLosses.dropna()



# printout

df_LowLosses
# load dataframe

df_Mfinalists = pd.read_csv('../input/ncaa-data/MNCAATourneyCompactResults.csv', low_memory = False)



#select rows with season value 2019

df_Mfinalists2019 = df_Mfinalists.loc[df_Mfinalists['Season'] == 2019]



# drop unnecessary columna

dropcolumns = ['LTeamID','WLoc', 'NumOT']

df_Mfinalists2019 = df_Mfinalists2019.drop(dropcolumns, axis=1)



# rename WTeamID column

df_Mfinalists2019 =  df_Mfinalists2019.rename(columns = {'WTeamID':'TeamID'}) 



# merge the dataframes

df_LowWins = pd.merge(df_LowMen19, df_Mfinalists2019, on='TeamID', how='left')



# drop unnecessary columna

dropcolumns = ['Season_y', 'DayNum']

df_LowWins = df_LowWins.drop(dropcolumns, axis=1)



# rename Season_x column

df_LowWins =  df_LowWins.rename(columns = {'Season_x':'Season'}) 



# drop columns with no value

df_LowWins = df_LowWins.dropna()



# printout

df_LowWins
# select only seeds 13-16

df_LowMen2 = df_SeedNames[(df_SeedNames['Seed']>=13)]



# select seasons 2015-2018

df_Low1518 = df_LowMen2[(df_LowMen2['Season']>=2015) & (df_LowMen2['Season']<=2018)]



# printout

df_Low1518.tail(8)
# load dataframe

df_Mfinalists = pd.read_csv('../input/ncaa-data/MNCAATourneyCompactResults.csv', low_memory = False)



# select seasons 2015-2019

df_Mfinalists1518 = df_Mfinalists[(df_Mfinalists['Season']>=2015) & (df_Mfinalists['Season']<=2018)]



# drop unnecessary columna

dropcolumns = ['LTeamID','WLoc', 'NumOT']

df_Mfinalists1518 = df_Mfinalists1518.drop(dropcolumns, axis=1)



# rename WTeamID column

df_Mfinalists1518 =  df_Mfinalists1518.rename(columns = {'WTeamID':'TeamID'}) 
# merge the dataframes

df_LowWins2 = pd.merge(df_Low1518, df_Mfinalists1518,  how='left', left_on=['Season','TeamID'], right_on = ['Season','TeamID'])



# drop unnecessary columna

dropcolumns = ['DayNum']

df_LowWins2 = df_LowWins2.drop(dropcolumns, axis=1)



# drop columns with no value

df_LowWins2 = df_LowWins2.dropna()



# make column values integers

df_LowWins2[['WScore', 'LScore']] = df_LowWins2[['WScore', 'LScore']].astype(int)



# printout

df_LowWins2
# select seeds 13-15

df_LowWins2 = df_LowWins2[(df_LowWins2['Seed']>=13) & (df_LowWins2['Seed']<=15)]



# printout

df_LowWins2
# select seasons 2015-2019

df_Low1519 = df_LowMen2[(df_LowMen2['Season']>=2015) & (df_LowMen2['Season']<=2019)]



# load dataframe

df_Mfinalists = pd.read_csv('../input/ncaa-data/MNCAATourneyCompactResults.csv', low_memory = False)



# select seasons 2015-2019

df_Mfinalists1519 = df_Mfinalists[(df_Mfinalists['Season']>=2015) & (df_Mfinalists['Season']<=2019)]



# drop unnecessary columna

dropcolumns = ['LTeamID','WLoc', 'NumOT']

df_Mfinalists1519 = df_Mfinalists1519.drop(dropcolumns, axis=1)



# rename WTeamID column

df_Mfinalists1519 =  df_Mfinalists1519.rename(columns = {'WTeamID':'TeamID'}) 



# merge the dataframes

df_LowWins2 = pd.merge(df_Low1519, df_Mfinalists1519,  how='left', left_on=['Season','TeamID'], right_on = ['Season','TeamID'])



# drop unnecessary columna

dropcolumns = ['DayNum']

df_LowWins2 = df_LowWins2.drop(dropcolumns, axis=1)



# drop columns with no value

df_LowWins2 = df_LowWins2.dropna()



# make column values integers

df_LowWins2[['WScore', 'LScore']] = df_LowWins2[['WScore', 'LScore']].astype(int)



# select seeds 13-15

df_LowWins2 = df_LowWins2[(df_LowWins2['Seed']>=13) & (df_LowWins2['Seed']<=15)]



# rename MTeamID column

df_classes = df_classes.rename(columns = {'MTeamID':'TeamID'}) 



# merge the dataframes

df_UpsetMascots = pd.merge(df_LowWins2, df_classes, on='TeamID', how='left')



# rename TeamName_x column

df_UpsetMascots = df_UpsetMascots.rename(columns = {'TeamName_x':'TeamName'}) 



# drop unnecessary columna

dropcolumns = ['WTeamID','TeamName_y']

df_UpsetMascots = df_UpsetMascots.drop(dropcolumns, axis=1)



# printout

df_UpsetMascots
# remove letters from Seeds column

df_w1519c = df_w1519b.replace(regex=['W','X','Y','Z','a','b'], value=' ')



# make Seeds column numeric

df_w1519c['Seed'] = pd.to_numeric(df_w1519c['Seed'])



# rename TeamID column

df_w1519c = df_w1519c.rename(columns = {'WTeamID':'TeamID'}) 



# load women's teams dataset

df_WTeams = pd.read_csv('../input/ncaa-data/WTeams.csv', low_memory = False)



# merge dataframes

df_WSeedNames = pd.merge(df_w1519c, df_WTeams, on='TeamID', how='left')



# select seeds 13-16

df_LowWomen = df_WSeedNames[(df_WSeedNames['Seed']>=13)]



#load dataframe

df_Wfinalists = pd.read_csv('../input/ncaa-data/WNCAATourneyCompactResults.csv', low_memory = False)



# select seasons 2015-2019

df_Wfinalists1519_ = df_Wfinalists[(df_Wfinalists['Season']>=2015) & (df_Wfinalists['Season']<=2019)]



# drop unnecessary columna

dropcolumns = ['DayNum','LTeamID','WLoc','NumOT']

df_Wfinalists1519_ = df_Wfinalists1519_.drop(dropcolumns, axis=1)



# rename TeamID column

df_Wfinalists1519_ =  df_Wfinalists1519_.rename(columns = {'WTeamID':'TeamID'}) 



# merge the dataframes

df_WLowWins2 = pd.merge(df_LowWomen, df_Wfinalists1519_,  how='left', left_on=['Season','TeamID'], right_on = ['Season','TeamID'])



# drop columns with no value

df_WLowWins2 = df_WLowWins2.dropna()



# printout

df_WLowWins2.head()
# load women's teams dataframe

df_WTeams = pd.read_csv('../input/ncaa-data/WTeams.csv', low_memory = False)



# merge dataframes

df_WSeedNames = pd.merge(df_w1519c, df_WTeams, on='TeamID', how='left')



# select seeds 11-12

df_LowWomen2 = df_WSeedNames[(df_WSeedNames['Seed']>=11) & (df_WSeedNames['Seed']<=12)]



# select seasons 2015-2019 for 11th and 12th seeded teams

df_WSeeds1519 = df_LowWomen2[(df_LowWomen2['Season']>=2015) & (df_LowWomen2['Season']<=2019)]



# load results dataframe

df_Wfinalists = pd.read_csv('../input/ncaa-data/WNCAATourneyCompactResults.csv', low_memory = False)



# select results from seasons 2015-2019

df_Wfinalists1519_ = df_Wfinalists[(df_Wfinalists['Season']>=2015) & (df_Wfinalists['Season']<=2019)]



# drop unnecessary columna

dropcolumns = ['DayNum','LTeamID','WLoc','NumOT']

df_Wfinalists1519_ = df_Wfinalists1519_.drop(dropcolumns, axis=1)



# rename TeamID column

df_Wfinalists1519_ =  df_Wfinalists1519_.rename(columns = {'WTeamID':'TeamID'}) 



# merge the dataframes

df_WLowWins3 = pd.merge(df_WSeeds1519, df_Wfinalists1519_,  how='left', left_on=['Season','TeamID'], right_on = ['Season','TeamID'])



# drop columns with no value

df_WLowWins3 = df_WLowWins3.dropna()



# printout

df_WLowWins3.head()
# load results dataframe

df_Mfinalists = pd.read_csv('../input/ncaa-data/MNCAATourneyCompactResults.csv', low_memory = False)



# select seasons 2015-2019

df_Mfinalists2019 = df_Mfinalists[(df_Mfinalists['Season']==2019)]



# drop unnecessary columna

dropcolumns = ['DayNum', 'WLoc', 'NumOT']

df_Mfinalists2019 = df_Mfinalists2019.drop(dropcolumns, axis=1)



# printout of last eight rows

df_Mfinalists2019.tail(8)
# select the four rows mentioned

df_ElEight19 = df_Mfinalists2019.loc[2244:2247]



# printout

df_ElEight19
# make lists of team IDs

EliteTeamsW = df_ElEight19.WTeamID.tolist()

EliteTeamsL = df_ElEight19.LTeamID.tolist()



# combine the two lists

EliteTeams19 = EliteTeamsW + EliteTeamsL



# convert list to dataframe

df_EliteMen = pd.DataFrame(EliteTeams19,columns=['TeamID'])



# load men's teams dataframe

df_MTeams = pd.read_csv('../input/ncaa-data/MTeams.csv', low_memory = False)



# merge dataframes

df_EliteMen2019 = pd.merge(df_EliteMen, df_MTeams, on='TeamID', how='left')



# drop unnecessary columna

dropcolumns = ['FirstD1Season','LastD1Season']

df_EliteMen2019 = df_EliteMen2019.drop(dropcolumns, axis=1)



# merge the dataframes

df_EliteMascots = pd.merge(df_EliteMen2019, df_classes, on='TeamID', how='left')



# rename TeamName_x column

df_EliteMascots = df_EliteMascots.rename(columns = {'TeamName_x':'TeamName'}) 



# drop unnecessary columna

dropcolumns = ['WTeamID','TeamName_y']

df_EliteMascots = df_EliteMascots.drop(dropcolumns, axis=1)



# printout

df_EliteMascots
# load results dataframe

df_Wfinalists = pd.read_csv('../input/ncaa-data/WNCAATourneyCompactResults.csv', low_memory = False)



# select seasons 2015-2019

df_Wfinalists2019 = df_Wfinalists[(df_Wfinalists['Season']==2019)]



# drop unnecessary columna

dropcolumns = ['DayNum', 'WLoc', 'NumOT']

df_Wfinalists2019 = df_Wfinalists2019.drop(dropcolumns, axis=1)



# show last eight rows

df_Wfinalists2019.tail(8)
# select the four rows mentioned

df_WElEight19 = df_Wfinalists2019.loc[1379:1382]



# make lists of team IDs

WEliteTeamsW = df_WElEight19.WTeamID.tolist()

WEliteTeamsL = df_WElEight19.LTeamID.tolist()



# combine the two lists

WEliteTeams19 = WEliteTeamsW + WEliteTeamsL



# convert list to dataframe

df_EliteWomen = pd.DataFrame(WEliteTeams19,columns=['TeamID'])



# load men's teams dataframe

df_WTeams = pd.read_csv('../input/ncaa-data/WTeams.csv', low_memory = False)



# merge dataframes

df_EliteWomen2019 = pd.merge(df_EliteWomen, df_WTeams, on='TeamID', how='left')



# rename team ID column

df_EliteWomen2019 = df_EliteWomen2019.rename(columns = {'TeamID':'WTeamID'}) 



# merge the dataframes

df_WEliteMascots = pd.merge(df_EliteWomen2019, df_classes, on='WTeamID', how='left')



# drop unnecessary columna

dropcolumns = ['TeamID','TeamName_y']

df_WEliteMascots = df_WEliteMascots.drop(dropcolumns, axis=1)



# rename TeamName_x and WTeamID columns

df_WEliteMascots = df_WEliteMascots.rename(columns = {'TeamName_x':'TeamName', 'WTeamID':'TeamID'}) 



# printout

df_WEliteMascots