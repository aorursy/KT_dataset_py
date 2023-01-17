# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
characters=pd.read_csv("../input/characters.csv")
comics=pd.read_csv("../input/comics.csv")
superheroes_power_matrix=pd.read_csv("../input/superheroes_power_matrix.csv")
charcters_stats=pd.read_csv("../input/charcters_stats.csv")
marvel_dc_characters=pd.read_excel("../input/marvel_dc_characters.xlsx")
marvel_characters_info=pd.read_csv("../input/marvel_characters_info.csv")
charactersToComics=pd.read_csv("../input/charactersToComics.csv")
# e-01
# question: Good versus Evil - Which group has more combined power?
# difficulty: easy
# datasets: charcters_stats.csv

##Group by alignment and aggregate total powers.
print(charcters_stats.groupby(by='Alignment').agg(['count','mean','sum'])['Total'])
print('Total power of good ones are greater than the bad ones')
print('But if we do this comparison by average power of good and bad ones; an average bad character has more power than a average good one.')

sns.boxplot(x='Alignment',
           y='Total',
           data=charcters_stats);
# e-02
# question:   Which alignment (good\bad) has higher avg speed?
# difficulty: easy
# datasets: charcters_stats

##Create a filter to find only good and bad ones
filterGoodAndBad=charcters_stats['Alignment'].isin(['good','bad'])

##Calculate average speed and find the fastest group.
SpeedChampionGroup=charcters_stats[filterGoodAndBad].groupby(by='Alignment').mean()['Speed'].idxmax()
print('Average speed is higher in {} ones'.format(SpeedChampionGroup))

# e-03
# question: How many superheros have more Intelligence then strength?
# difficulty: easy
# datasets: charcters_stats

## Create a filter to find the smarter heroes.
SmarterHeroesFilter=charcters_stats['Intelligence']>charcters_stats['Strength']
SmarterHeroes=charcters_stats[SmarterHeroesFilter]['Name']
print('There are {} heroes have more Intelligence then strength'.format(SmarterHeroes.size))
print('Here are some of them:')
SmarterHeroes.head()

# e-04
# question: Show the distribution of Total (all the powers combined) for the good and the evil.
# difficulty: easy
# datasets: charcters_stats

#Creating two filters to find good and bad characters.
filterGood=charcters_stats['Alignment'].isin(['good'])
filterBad=charcters_stats['Alignment'].isin(['bad'])

#Creating a figure for 2 plots
f,(ax1,ax2)=plt.subplots(2,1,figsize=(7,7), sharex=True, sharey=True)

#Plot the data and show the distribution
sns.distplot(charcters_stats[filterGood]['Total'],ax=ax1)
ax1.set_ylabel("Good")
sns.distplot(charcters_stats[filterBad]['Total'],ax=ax2)
ax2.set_ylabel("Bad")
# e-05
# question: How many comics with 7 or more character published each year? (show on graph)
# difficulty: easy

## Creating a filter to find the comics with least 8 characters in 
comicsWith7orMoreCharacterFilter=charactersToComics.groupby(by='comicID').count()['characterID']>7
## Applying the filter and find comics with 7+ characters in it
comicsWith7orMoreCharacterData=charactersToComics.groupby(by='comicID').count()[comicsWith7orMoreCharacterFilter]

##Finding year of the comics
### First, creating a function to extract the year from comics title column
def extractYear(txt):
    startChr=txt.find('(')+1
    endChr=txt.find(')')
    try:
        result=int(txt[startChr:endChr])
        return result
    except:
        pass
    
###Creating a copy of comics data frame
comicsWyear=comics.copy()

###applying extractYear function and creating a new column that contains year of the comics
comicsWyear['year']=comics['title'].apply(extractYear)

### creating a bar chart
comicsWyear.groupby(by='year').count()['comicID'].plot(kind='bar',figsize=(18,12));

del comicsWyear
# m-01
# question: Show 5 top comics with top participants on a plot bar.
# difficulty: medium
# datasets: comics,characters,comics_to_characters


## First, find number of participants in comics
comicsAndNumberOfParticipants=charactersToComics.groupby(by='comicID').count().sort_values(by='characterID',ascending=False)
##Find top 5 comics with most participants
comicsWithMostParticipants=pd.DataFrame(comicsAndNumberOfParticipants.head(5))
comicsWithMostParticipants.columns=['NumberOfParticipants']

##Use comics dataframe to find comics' names
comicsWithMostParticipantsAndNames=pd.merge(left=comicsWithMostParticipants,
                                    right=comics,
                                    on='comicID'
                                   );
##plot it
sns.barplot(y='title',
           x='NumberOfParticipants',
           data=comicsWithMostParticipantsAndNames
           );

# delete temporary objects
del [comicsAndNumberOfParticipants,comicsWithMostParticipants,comicsWithMostParticipantsAndNames]
# m-02
# question: Unmatched rivals - show for each super hero the number of vilans that stronger then him/her
# difficulty: medium-hard
# datasets: charcters_stats.csv

#First create hero and villain dataframe by using Alignment feature
heroFilter=charcters_stats['Alignment'].isin(['good'])
heroesData=charcters_stats[heroFilter][['Name','Total']]

villainFilter=charcters_stats['Alignment'].isin(['bad'])
villainData=charcters_stats[villainFilter][['Name','Total']]

#You will need a dummy column for merging dataframes
heroesData['key']=0
villainData['key']=0

#You merge dataframes on dummy columns, key. It is not an efficient way.
heroesVillainDataXJoined=heroesData.merge(villainData,how='outer',left_on=['key'],right_on=['key'])

#You create a filter to find stronger ones
findStrongerOnesFilter=heroesVillainDataXJoined['Total_y']>heroesVillainDataXJoined['Total_x']
findStrongerOnes=heroesVillainDataXJoined[findStrongerOnesFilter]

#You count and list the number of villains that are stronger than a hero.
NumberStrongerOnes=findStrongerOnes.groupby('Name_x').count()['Name_y']
print(NumberStrongerOnes.head(15))

# delete temporary objects
del [heroFilter,heroesData,villainFilter,villainData,heroesVillainDataXJoined,findStrongerOnesFilter,findStrongerOnes]

# h-01
# question: Show pairs of characters that always appear together. rank them by number of appearances
# difficulty: hard - very hard
# datasets: comics,characters,comics_to_characters

#Characters to comics data's used.
#You merge characters to comics data so you can calculate how many times are pair of characters together.

dataTemp1=pd.merge(left=charactersToComics,
                    right=charactersToComics,
                    how='inner',
                    on='comicID')

#After merging, characters are matched with other characters and their own. You delete rows which characters matched with themselves.
eliminateEqualOnesFilter=dataTemp1['characterID_x']!=dataTemp1['characterID_y']
dataTemp2=dataTemp1[eliminateEqualOnesFilter]

#Grouping the data by characters and counting that how many comics that they are together in
dataTemp3=dataTemp2.groupby(by=['characterID_x','characterID_y']).count()

#Finding characters that are together the most
dataTemp5=dataTemp3.idxmax().get_values()
numberOfTimesTheyAreTogether=dataTemp3['comicID'].max()

# Using characters data for character's name
characterId_1=dataTemp5[0][0]
characterName_1=characters['name'][characters['characterID'].isin([characterId_1])].get_values()[0]

characterId_2=dataTemp5[0][1]
characterName_2=characters[characters['characterID'].isin([characterId_2])]['name'].get_values()[0]
print('Most frequent pair of characters are {} and {}. There are in same comics {} of times.'.format(characterName_1,characterName_2,numberOfTimesTheyAreTogether))

# h-02
# question: Unmatched rivals - show for each super hero , all the names of the  vilans that stronger then him/her
# difficulty: hard
# datasets: charcters_stats.csv

# BONUS
# question: Unmatched rivals - find an informative way to visualize the results you got

# This question is same as m-02. But this time you try to visualize the result.
#First creating 2 dataframes, hero and villain, by using Alignment feature, like you did in m-02.
heroFilter=charcters_stats['Alignment'].isin(['good'])
heroesData=charcters_stats[heroFilter][['Name','Total']]
villainFilter=charcters_stats['Alignment'].isin(['bad'])
villainData=charcters_stats[villainFilter][['Name','Total']]

#You need a dummy column for merging dataframes
heroesData['key']=0
villainData['key']=0

#Merging dataframes on dummy columns, key. Maybe it is not the efficient way.
heroesVillainDataXJoined=heroesData.merge(villainData,how='outer',left_on=['key'],right_on=['key'])

#Creating two columns that represents the difference of the powers if the difference is greater than 0.
heroesVillainDataXJoined['PowerDifference']=heroesVillainDataXJoined['Total_y']-heroesVillainDataXJoined['Total_x']

#Keep only columns you need
heroesVillainDataXJoined=heroesVillainDataXJoined[['Name_x','Name_y','PowerDifference']]
#Checking if there is any duplicate value.If any,the first one is kept and rest is dropped. You get an error while you visualize them If duplicate value exists. 
heroesVillainDataXJoined=heroesVillainDataXJoined.drop_duplicates(subset=['Name_x','Name_y'],keep='first')
heroesVillainDataXJoined['IsVillainStronger']=heroesVillainDataXJoined['PowerDifference']<0
#Creating 2 heat maps that represents how much stronger that villains are than each heroes.
##First, heat map without any filter
heatmap_data=heroesVillainDataXJoined.pivot('Name_x','Name_y','PowerDifference')
fig=plt.figure()
fig.set_figheight(40)
fig.set_figwidth(30)

ax1=fig.add_subplot(2,1,1)
sns.heatmap(heatmap_data,
            ax=ax1,
           cmap='bwr')
ax1.set_title('Heatmap without any filter')

##Heat map with the stronger filter
heatmap_filter=heroesVillainDataXJoined.pivot('Name_x','Name_y','IsVillainStronger')

ax2=fig.add_subplot(2,1,2)
sns.heatmap(heatmap_data,
            mask=heatmap_filter,
            vmin=0,
            cmap='Reds',
            ax=ax2)
ax2.set_title('Heatmap with the filter')
# e-06
# question: How has more characters DC or Marvel?
# difficulty: easy
# datasets: comics,characters,comics_to_characters
# e-07
# question: Who has higher representation of female heros DC or Marvel?
# difficulty: easy
# datasets: marvel_dc_characters
# e-08
# question: Who has higher representation of black skined heros DC or Marvel?
# difficulty: easy
# datasets: marvel_dc_characters
# e-09
# question: Show how common is each trait in 'superheroes_power_matrix.csv'.
# difficulty: easy
# datasets: superheroes_power_matrix
# e-10
# question: Show the hight distrebution for the characters of 'Marvel Comics' (from 'marvel_characters_info.csv').
# difficulty: easy
# datasets: 
# e-11
# question: Show the distrebution of apperences.
# difficulty: easy
# datasets: marvel_dc_characters.csv
# e-12
# question: Show the distrebution of eye colors.
# difficulty: easy
# datasets: marvel_dc_characters.csv
# e-13
# question: How many characters apperred only once?
# difficulty: easy
# datasets: marvel_dc_characters.csv
# e-14
# question: How many characters died in thair first apperance (have one apperance and are deceased)?
# difficulty: easy
# datasets: marvel_dc_characters.csv
# e-15
# question:   Display a pie chart of the 10 most common hair styles
# difficulty: easy
# datasets: marvel_characters_info


# e-16
# question: Display the average height
# difficulty: easy
# datasets: marvel_characters_info

# e-17
# find the comic with most characters. display the comics name and the name of the characters
# difficulty: easy

# e-18
# the oldest character of both universes
# difficulty: easy

# e-19
# we want to build the master group to fight evil, kind of an avengers 2.0, but only better,
# lets select the captain, the one with the most total stats  (obviously his Alignment must be good to fight evil)
# level: easy

# e-20
# People will pay big money for original vintage comic books, retrive all first issue comic books
# level: easy
# datasets: comics_characters

# e-21
# On the other hand, long lasting series are great as well :), retrive the comic book with the biggest issue number
#level: easy
# datasets: comics_characters


# e-22
# It's the holiday season, and to celebrate marvel usually comes out with holiday special comic books, 
# retrive all  holiday special comic books (the word 'Holiday' will appeer in the title)
# level: easy

# e-23
# What's the mean intelligence of the superheroes who have a 'True' value in the power matrix and the same for the superheroes who have a 'False' value?
# difficulty: easy


#Medium Questions

# m-03
# question: Weak point - for each vilan, show his weakest characteristic.
# difficulty: medium
# datasets: charcters_stats
# m-04
# question: Who can beat me? - for each vilan, show how many superheros can defeat them (compare by total score)
# difficulty: medium
# datasets: charcters_stats
# m-05
# question: Display box plot summarizing the next statistics:
# Height, Weight, Intelligence, Strength, Speed, Durability, Power, Combat
# difficulty: medium
# datasets: marvel_characters_info, characters_stats
# m-06
# after you found it display the characters the participate it in
# difficulty: medium

# m-07
# A great team needs great diversity, and to be great at everything, get the best hero at each statistical category
# level: easy - medium

# m-08
# Is your strngth and intelligence related?.
# Show a scatter chart where the x axis is stength, and the y axis is intelligence, scatter heros and villans as two different color dots
# level: easy - medium


# m-09
# To truly be a great superhero, you can't be a one trick pony, you need to posess multipule abilities. Create a series of every superhero and how many different abilities they posess, in descending order
# level: medium

# m-10
# Create a serires that counts the number of comic book appeerences for each hero
# Bonus: show the top 10 heros in a pie chart

#level: easy - medium


# m-11
# Pick any hero from the previous question and list all the comic book titles that he appeared in
#level: medium


# m-12
# It's the holiday season once again, since we already have a list of all holiday comics, 
# retrive all heros who have participated in a holiday comic book

# level: easy - medium


# h-03
#find the heroes that has the maximum amount of abilities(more True values than anyone)
# difficulty: hard (not that hard)

# h-04
# Two of the most iconic marvel superheros, Iron Man and Captain America, appeer together quite offten. 
# see if you can get the ammount of comic books they both appear in

# level: medium - hard


# h-05
# Now that we know how many comic books both of those guys have appeared together at, are they the best power duo in the marvel universe?.
# craete a series with a multi index of 2 superheros(name1,name2) and count for each of them the ammount of comic books they have been in together in, order by that ammount in a descending order

# level: really hard :)
