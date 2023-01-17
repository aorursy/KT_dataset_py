# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt
arquivo = '../input/data.csv'

dataset = pd.read_csv(arquivo, sep = ',', header=0)
def alteraPalavra(string):

    if string.endswith('K') or string.endswith('M'):

        string = string[:-1]

    string = string[1:]

    return string



def transStr(object):

    return str(object)



def transFloat(string):

    return float(string)



def containsMorN(s):

    if "M" in s:

        return 1

    else:

        return 0.001

#Checking if the file was imported properly

dataset.head()
#treating missing values on alternative position overall of players, loans and market value

dataset['Loaned From'].fillna(0, inplace = True)

dataset['LS'].fillna(0, inplace = True)

dataset['ST'].fillna(0, inplace = True)

dataset['RS'].fillna(0, inplace = True)

dataset['LW'].fillna(0, inplace = True)

dataset['LF'].fillna(0, inplace = True)

dataset['CF'].fillna(0, inplace = True)

dataset['RF'].fillna(0, inplace = True)

dataset['RW'].fillna(0, inplace = True)

dataset['LAM'].fillna(0, inplace = True)

dataset['CAM'].fillna(0, inplace = True)

dataset['RAM'].fillna(0, inplace = True)

dataset['LM'].fillna(0, inplace = True)

dataset['LCM'].fillna(0, inplace = True)

dataset['CM'].fillna(0, inplace = True)

dataset['RCM'].fillna(0, inplace = True)

dataset['RM'].fillna(0, inplace = True)

dataset['LWB'].fillna(0, inplace = True)

dataset['LDM'].fillna(0, inplace = True)

dataset['CDM'].fillna(0, inplace = True)

dataset['RDM'].fillna(0, inplace = True)

dataset['RWB'].fillna(0, inplace = True)

dataset['LB'].fillna(0, inplace = True)

dataset['LCB'].fillna(0, inplace = True)

dataset['CB'].fillna(0, inplace = True)

dataset['RCB'].fillna(0, inplace = True)

dataset['RB'].fillna(0, inplace = True)

dataset['Release Clause'].fillna(100, inplace = True)
#Pre-processing value and wage data

dataset['valor'] = (dataset['Value'].apply(transStr))

dataset['grauValor'] = (dataset['valor'].apply(containsMorN))

dataset['valor'] = (dataset['valor'].apply(alteraPalavra))

dataset['valor'] = (dataset['valor'].apply(transFloat))

dataset['valor'] = (dataset['valor'] * dataset['grauValor'])

dataset['valor'].fillna(0, inplace = True)



dataset['Wage'] = (dataset['Wage'].apply(transStr))

dataset['Wage'] = (dataset['Wage'].apply(alteraPalavra))

dataset['Wage'] = (dataset['Wage'].apply(transFloat))
%matplotlib inline

plt.rcParams['figure.figsize'] = (20,10)

plt.style.use("ggplot")

dataset.plot(x = 'Overall', y = 'valor', kind='scatter', title= 'Overall x Value in Millions of Euros')
%matplotlib inline

plt.rcParams['figure.figsize'] = (20,10)

dataset.plot(x = 'Age', y = 'valor', kind='scatter', title= 'Age x Value in Millions of Euros')
%matplotlib inline

plt.rcParams['figure.figsize'] = (20,10)

dataset.plot(x = 'Age', y = 'Overall', kind='scatter', title= 'Age x Overall')
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,10)



plt.subplot(1,3,1, title = 'All players Overall Distribution') #numero de linhas, numero de colunas, qual utilizar

plt.boxplot(dataset['Overall'])



dataMaior34 = dataset.loc[dataset['Age'] >= 34]



dataMenor34 = dataset.loc[dataset['Age'] < 34]



plt.subplot(1,3,2, title = 'Players younger than 34 Overall Distribution')

plt.boxplot(dataMaior34['Overall'])



plt.subplot(1,3,3, title = 'Players older than 33 Overall Distribution')

plt.boxplot(dataMenor34['Overall'])
dataset['Overall'].describe()
dataMaior34['Overall'].describe()
dataMenor34['Overall'].describe()
dataset['Wage'].describe()
dataMaior34['Wage'].describe()
dataMenor34['Wage'].describe()
dataset['valor'].describe()
dataMaior34['valor'].describe()
dataMenor34['valor'].describe()
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,15)



plt.subplot(3,3,1, title = 'Acceleration x Market Value')

plt.scatter(dataset['Acceleration'], dataset['valor'], c = '#8e0000')



plt.subplot(3,3,2, title = 'Sprint Speed x Market Value')

plt.scatter(dataset['SprintSpeed'], dataset['valor'], c = '#8ed000')



plt.subplot(3,3,3, title = 'Agility x Market Value') 

plt.scatter(dataset['Agility'], dataset['valor'], c = '#8e6c82')



plt.subplot(3,3,4, title = 'Balance x Market Value')

plt.scatter(dataset['Balance'], dataset['valor'], c = '#40c6e0')



plt.subplot(3,3,5, title = 'Reactions x Market Value')

plt.scatter(dataset['Reactions'], dataset['valor'], c = '#ffff00')



plt.subplot(3,3,6, title = 'Jumping x Market Value')

plt.scatter(dataset['Jumping'], dataset['valor'], c = '#4f3e00')



plt.subplot(3,3,7, title = 'Stamina x Market Value')

plt.scatter(dataset['Stamina'], dataset['valor'], c = '#ed233d')



plt.subplot(3,3,8, title = 'Strength x Market Value')

plt.scatter(dataset['Strength'], dataset['valor'], c = '#39843d')
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,30)



plt.subplot(6,3,1, title = 'Finishing x Market Value') 

plt.scatter(dataset['Finishing'], dataset['valor'], c = '#8e0000')



plt.subplot(6,3,2, title = 'Long Shots x Market Value')

plt.scatter(dataset['LongShots'], dataset['valor'], c = '#8ed000')



plt.subplot(6,3,3, title = 'Penalties x Market Value') 

plt.scatter(dataset['Penalties'], dataset['valor'], c= '#8e6c82')



plt.subplot(6,3,4, title = 'Shot Power x Market Value')

plt.scatter(dataset['ShotPower'], dataset['valor'], c = '#40c6e0')



plt.subplot(6,3,5, title = 'Volleys x Market Value')

plt.scatter(dataset['Volleys'], dataset['valor'], c = '#ffff00')



plt.subplot(6,3,6, title = 'Crossing x Market Value')

plt.scatter(dataset['Crossing'], dataset['valor'], c = '#4f3e00')



plt.subplot(6,3,7, title = 'Curve x Market Value')

plt.scatter(dataset['Curve'], dataset['valor'], c = '#ed233d')



plt.subplot(6,3,8, title = 'FK Accuracy x Market Value')

plt.scatter(dataset['FKAccuracy'], dataset['valor'], c = '#39843d')



plt.subplot(6,3,9, title = 'Ball Control x Market Value')

plt.scatter(dataset['BallControl'], dataset['valor'], c = '#ffcc99')



plt.subplot(6,3,10, title = 'Dribbling x Market Value')

plt.scatter(dataset['Dribbling'], dataset['valor'], c = '#ff00ff')



plt.subplot(6,3,11, title = 'Heading Accuracy x Market Value')

plt.scatter(dataset['HeadingAccuracy'], dataset['valor'], c = '#000099')



plt.subplot(6,3,12, title = 'Marking x Market Value')

plt.scatter(dataset['Marking'], dataset['valor'], c = '#cccc00')



plt.subplot(6,3,13, title = 'Sliding Tackle x Market Value')

plt.scatter(dataset['SlidingTackle'], dataset['valor'], c = '#999966')



plt.subplot(6,3,14, title = 'Standing tackle x Market Value')

plt.scatter(dataset['StandingTackle'], dataset['valor'], c = '#ff3300')



plt.subplot(6,3,15, title = 'Long Passing x Market Value')

plt.scatter(dataset['LongPassing'], dataset['valor'], c = '#cc00ff')



plt.subplot(6,3,16, title = 'Short Passing x Market Value')

plt.scatter(dataset['ShortPassing'], dataset['valor'], c = '#99ffcc')
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,10)



plt.subplot(2,3,1, title = 'Positioning x Market Value') #numero de linhas, numero de colunas, qual utilizar

plt.scatter(dataset['Positioning'], dataset['valor'], c = '#8e0000')



plt.subplot(2,3,2, title = 'Vision x Market Value')

plt.scatter(dataset['Vision'], dataset['valor'], c = '#8ed000')



plt.subplot(2,3,3, title = 'Composure x Market Value') #numero de linhas, numero de colunas, qual utilizar

plt.scatter(dataset['Composure'], dataset['valor'], c= '#8e6c82')



plt.subplot(2,3,4, title = 'Interceptions x Market Value')

plt.scatter(dataset['Interceptions'], dataset['valor'], c = '#40c6e0')



plt.subplot(2,3,5, title = 'Aggression x Market Value')

plt.scatter(dataset['Aggression'], dataset['valor'], c = '#ffff00')
dataset[['valor', 'Reactions', 'LongPassing', 'ShortPassing', 'BallControl', 'Dribbling', 'Vision', 'Composure']].corr('spearman')
mbrazil = dataset.loc[dataset['Nationality'] != 'Brazil']

mgermany = mbrazil.loc[mbrazil['Nationality'] != 'Germany']

mitaly = mgermany.loc[mgermany['Nationality'] != 'Italy']

margentina = mitaly.loc[mitaly['Nationality'] != 'Argentina']

others = margentina.loc[margentina['Nationality'] != 'France']
brazil = dataset.loc[dataset['Nationality'] == 'Brazil']

germany = dataset.loc[dataset['Nationality'] == 'Germany']

italy = dataset.loc[dataset['Nationality'] == 'Italy']

argentina = dataset.loc[dataset['Nationality'] == 'Argentina']

france = dataset.loc[dataset['Nationality'] == 'France']
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,15)



plt.subplot(3,3,1, title = 'All Players Overall x Market Value') 

plt.scatter(dataset['Overall'], dataset['valor'], c = '#8e0000')



plt.subplot(3,3,2, title = 'Brazilians Overall x Market Value' )

plt.scatter(brazil['Overall'], brazil['valor'], c = '#ffff00')



plt.subplot(3,3,3, title = 'Germans Overall x Market Value')

plt.scatter(germany['Overall'], germany['valor'], c= '#1f1f14')



plt.subplot(3,3,4, title = 'Italians Overall x Market Value')

plt.scatter(italy['Overall'], italy['valor'], c = '#0033cc')



plt.subplot(3,3,5, title = 'Argentines Overall x Market Value')

plt.scatter(argentina['Overall'], argentina['valor'], c = '#99b3ff')



plt.subplot(3,3,6, title = 'French Overall x Market Value')

plt.scatter(france['Overall'], france['valor'], c = '#6600cc')



plt.subplot(3,3,7, title = 'Other Countries Overall x Market Value')

plt.scatter(others['Overall'], others['valor'], c = '#65d741')
brazil = brazil.loc[(brazil['Overall'] >= 70) & (brazil['Overall'] <=80)]

germany = germany.loc[(germany['Overall'] >= 70) & (germany['Overall'] <=80)]

italy = italy.loc[(italy['Overall'] >= 70) & (italy['Overall'] <=80)]

argentina = argentina.loc[(argentina['Overall'] >= 70) & (argentina['Overall'] <=80)]

france = france.loc[(france['Overall'] >= 70) & (france['Overall'] <=80)]

others = others.loc[(others['Overall'] >= 70) & (others['Overall'] <=80)]

mediumdataset = dataset.loc[(dataset['Overall'] >= 70) & (dataset['Overall'] <=80)]
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,15)



plt.subplot(3,3,1, title = 'All Players Overall x Market Value') 

plt.scatter(mediumdataset['Overall'], mediumdataset['valor'], c = '#8e0000')



plt.subplot(3,3,2, title = 'Brazilians Overall x Market Value' )

plt.scatter(brazil['Overall'], brazil['valor'], c = '#ffff00')



plt.subplot(3,3,3, title = 'Germans Overall x Market Value')

plt.scatter(germany['Overall'], germany['valor'], c= '#1f1f14')



plt.subplot(3,3,4, title = 'Italians Overall x Market Value')

plt.scatter(italy['Overall'], italy['valor'], c = '#0033cc')



plt.subplot(3,3,5, title = 'Argentines Overall x Market Value')

plt.scatter(argentina['Overall'], argentina['valor'], c = '#99b3ff')



plt.subplot(3,3,6, title = 'French Overall x Market Value')

plt.scatter(france['Overall'], france['valor'], c = '#6600cc')



plt.subplot(3,3,7, title = 'Other Countries Overall x Market Value')

plt.scatter(others['Overall'], others['valor'], c = '#65d741')
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,15)



plt.subplot(3,3,1, title = 'All Players Overall Histogram') 

plt.hist(mediumdataset['Overall'], bins = 11, color = '#8e0000')



plt.subplot(3,3,2, title = 'Brazilians Overall Histogram')

plt.hist(brazil['Overall'], bins = 11,color = '#ffff00')



plt.subplot(3,3,3, title = 'Germans Overall Histogram')

plt.hist(germany['Overall'], bins = 11,color= '#1f1f14')



plt.subplot(3,3,4, title = 'Italians Overall Histogram')

plt.hist(italy['Overall'],bins = 11, color = '#0033cc')



plt.subplot(3,3,5, title = 'Argentines Overall Histogram')

plt.hist(argentina['Overall'], bins = 11,color = '#99b3ff')



plt.subplot(3,3,6, title = 'French Overall Histogram')

plt.hist(france['Overall'], bins = 11, color = '#6600cc')



plt.subplot(3,3,7, title = 'Other Countries Overall Histogram')

plt.hist(others['Overall'], bins = 11, color = '#65d741')
%matplotlib inline



plt.rcParams['figure.figsize'] = (10,8)



overallmean = []

overallmean.append(mediumdataset['Overall'].mean())

overallmean.append(brazil['Overall'].mean())

overallmean.append(germany['Overall'].mean())

overallmean.append(italy['Overall'].mean())

overallmean.append(argentina['Overall'].mean())

overallmean.append(france['Overall'].mean())

overallmean.append(others['Overall'].mean())



overallstd = []

overallstd.append(mediumdataset['Overall'].std())

overallstd.append(brazil['Overall'].std())

overallstd.append(germany['Overall'].std())

overallstd.append(italy['Overall'].std())

overallstd.append(argentina['Overall'].std())

overallstd.append(france['Overall'].std())

overallstd.append(others['Overall'].std())



labels = ['All players', 'Brazil', 'Germany', 'Italy', 'Argentina', 'France', 'Other countries']



colors = ['#8e0000', '#ffff00', '#1f1f14', '#0033cc', '#99b3ff', '#6600cc', '#65d741']



plt.bar(labels, overallmean, yerr = overallstd, color = colors)
%matplotlib inline



plt.rcParams['figure.figsize'] = (10,8)



valuemean = []

valuemean.append(mediumdataset['valor'].mean())

valuemean.append(brazil['valor'].mean())

valuemean.append(germany['valor'].mean())

valuemean.append(italy['valor'].mean())

valuemean.append(argentina['valor'].mean())

valuemean.append(france['valor'].mean())

valuemean.append(others['valor'].mean())



valuestd = []

valuestd.append(mediumdataset['valor'].std())

valuestd.append(brazil['valor'].std())

valuestd.append(germany['valor'].std())

valuestd.append(italy['valor'].std())

valuestd.append(argentina['valor'].std())

valuestd.append(france['valor'].std())

valuestd.append(others['valor'].std())



labels = ['All players', 'Brazil', 'Germany', 'Italy', 'Argentina', 'France', 'Other countries']



colors = ['#8e0000', '#ffff00', '#1f1f14', '#0033cc', '#99b3ff', '#6600cc', '#65d741']



plt.bar(labels, valuemean, yerr = valuestd, color = colors)
def definefieldpart(p):

    if p == 'GK':

        return 0

    elif p == 'RB' or p == 'CB' or p == 'LB' or p == 'RWB' or p == 'LWB' or p == 'LCB' or p == 'RCB':

        return 1

    elif p == 'CDM' or p == 'CM' or p == 'CAM' or p == 'RM' or p == 'LM' or p == 'RW' or p == 'LW' or p == 'LAM' or p == 'RAM' or p == 'RCM' or p == 'RDM' or p == 'LDM':

        return 2

    else:

        return 3
dataset['fieldpart'] = (dataset['Position'].apply(definefieldpart))

dataset.head()
goalkeepers = dataset.loc[dataset['fieldpart'] == 0]

defense = dataset.loc[dataset['fieldpart'] == 1]

midfielders = dataset.loc[dataset['fieldpart'] == 2]

attack = dataset.loc[dataset['fieldpart'] == 3]
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,15)



plt.subplot(2,2,1, title = 'Goalkeepers Overall Histogram') 

plt.hist(goalkeepers['Overall'], bins = 45, color = '#8e0000')



plt.subplot(2,2,2, title = 'Defenders Overall Histogram')

plt.hist(defense['Overall'], bins = 45,color = '#ffff00')



plt.subplot(2,2,3, title = 'Midfielders Overall Histogram')

plt.hist(midfielders['Overall'], bins = 45,color= '#1f1f14')



plt.subplot(2,2,4, title = 'Attackers Overall Histogram')

plt.hist(attack['Overall'],bins = 45, color = '#0033cc')



%matplotlib inline



plt.rcParams['figure.figsize'] = (20,15)



plt.subplot(2,2,1, title ='Goalkeepers Scatter distribution') 

plt.scatter(goalkeepers['Overall'], goalkeepers['valor'], c = '#8e0000')



plt.subplot(2,2,2, title ='Defensors Scatter distribution')

plt.scatter(defense['Overall'], defense['valor'],c = '#ffff00')



plt.subplot(2,2,3, title ='Midfielders Scatter distribution')

plt.scatter(midfielders['Overall'], midfielders['valor'],c= '#1f1f14')



plt.subplot(2,2,4, title ='Attackers Scatter distribution')

plt.scatter(attack['Overall'], attack['valor'], c = '#0033cc')
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,15)



gkb50 = goalkeepers.loc[goalkeepers['Overall'] < 50]

gkb5055 = goalkeepers.loc[(goalkeepers['Overall'] >=50) & (goalkeepers['Overall'] < 55)]

gkb5560 = goalkeepers.loc[(goalkeepers['Overall'] >=55) & (goalkeepers['Overall'] < 60)]

gkb6065 = goalkeepers.loc[(goalkeepers['Overall'] >=60) & (goalkeepers['Overall'] < 65)]

gkb6570 = goalkeepers.loc[(goalkeepers['Overall'] >=65) & (goalkeepers['Overall'] < 70)]

gkb7075 = goalkeepers.loc[(goalkeepers['Overall'] >=70) & (goalkeepers['Overall'] < 75)]

gkb7580 = goalkeepers.loc[(goalkeepers['Overall'] >=75) & (goalkeepers['Overall'] < 80)]

gkb8085 = goalkeepers.loc[(goalkeepers['Overall'] >=80) & (goalkeepers['Overall'] < 85)]

gkb8590 = goalkeepers.loc[(goalkeepers['Overall'] >=85) & (goalkeepers['Overall'] < 90)]

gka90 = goalkeepers.loc[goalkeepers['Overall'] >= 90]



gkmean = []

gkmean.append(gkb50['valor'].mean())

gkmean.append(gkb5055['valor'].mean())

gkmean.append(gkb5560['valor'].mean())

gkmean.append(gkb6065['valor'].mean())

gkmean.append(gkb6570['valor'].mean())

gkmean.append(gkb7075['valor'].mean())

gkmean.append(gkb7580['valor'].mean())

gkmean.append(gkb8085['valor'].mean())

gkmean.append(gkb8590['valor'].mean())

gkmean.append(gka90['valor'].mean())



gkvaluestd = []

gkvaluestd.append(gkb50['valor'].std())

gkvaluestd.append(gkb5055['valor'].std())

gkvaluestd.append(gkb5560['valor'].std())

gkvaluestd.append(gkb6065['valor'].std())

gkvaluestd.append(gkb6570['valor'].std())

gkvaluestd.append(gkb7075['valor'].std())

gkvaluestd.append(gkb7580['valor'].std())

gkvaluestd.append(gkb8085['valor'].std())

gkvaluestd.append(gkb8590['valor'].std())

gkvaluestd.append(gka90['valor'].std())



labels = ['< 50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '>90']



colors = ['#8e0000', '#ffff00', '#1f1f14', '#0033cc', '#99b3ff', '#6600cc', '#65d741' , "#46a345", "#985fff"]



plt.subplot(2,2,1, title = 'Goalkeepers Market Value Means') 

plt.bar(labels, gkmean, yerr = gkvaluestd, color = colors)



db50 = defense.loc[defense['Overall'] < 50]

db5055 = defense.loc[(defense['Overall'] >=50) & (defense['Overall'] < 55)]

db5560 = defense.loc[(defense['Overall'] >=55) & (defense['Overall'] < 60)]

db6065 = defense.loc[(defense['Overall'] >=60) & (defense['Overall'] < 65)]

db6570 = defense.loc[(defense['Overall'] >=65) & (defense['Overall'] < 70)]

db7075 = defense.loc[(defense['Overall'] >=70) & (defense['Overall'] < 75)]

db7580 = defense.loc[(defense['Overall'] >=75) & (defense['Overall'] < 80)]

db8085 = defense.loc[(defense['Overall'] >=80) & (defense['Overall'] < 85)]

db8590 = defense.loc[(defense['Overall'] >=85) & (defense['Overall'] < 90)]

da90 = defense.loc[defense['Overall'] >= 90]



dmean = []

dmean.append(db50['valor'].mean())

dmean.append(db5055['valor'].mean())

dmean.append(db5560['valor'].mean())

dmean.append(db6065['valor'].mean())

dmean.append(db6570['valor'].mean())

dmean.append(db7075['valor'].mean())

dmean.append(db7580['valor'].mean())

dmean.append(db8085['valor'].mean())

dmean.append(db8590['valor'].mean())

dmean.append(da90['valor'].mean())



dvaluestd = []

dvaluestd.append(db50['valor'].std())

dvaluestd.append(db5055['valor'].std())

dvaluestd.append(db5560['valor'].std())

dvaluestd.append(db6065['valor'].std())

dvaluestd.append(db6570['valor'].std())

dvaluestd.append(db7075['valor'].std())

dvaluestd.append(db7580['valor'].std())

dvaluestd.append(db8085['valor'].std())

dvaluestd.append(db8590['valor'].std())

dvaluestd.append(da90['valor'].std())



plt.subplot(2,2,2, title = 'Defensors Market Value Means') 

plt.bar(labels, dmean, yerr = dvaluestd, color = colors)



mb50 = midfielders.loc[midfielders['Overall'] < 50]

mb5055 = midfielders.loc[(midfielders['Overall'] >=50) & (midfielders['Overall'] < 55)]

mb5560 = midfielders.loc[(midfielders['Overall'] >=55) & (midfielders['Overall'] < 60)]

mb6065 = midfielders.loc[(midfielders['Overall'] >=60) & (midfielders['Overall'] < 65)]

mb6570 = midfielders.loc[(midfielders['Overall'] >=65) & (midfielders['Overall'] < 70)]

mb7075 = midfielders.loc[(midfielders['Overall'] >=70) & (midfielders['Overall'] < 75)]

mb7580 = midfielders.loc[(midfielders['Overall'] >=75) & (midfielders['Overall'] < 80)]

mb8085 = midfielders.loc[(midfielders['Overall'] >=80) & (midfielders['Overall'] < 85)]

mb8590 = midfielders.loc[(midfielders['Overall'] >=85) & (midfielders['Overall'] < 90)]

ma90 = midfielders.loc[midfielders['Overall'] >= 90]



mmean = []

mmean.append(mb50['valor'].mean())

mmean.append(mb5055['valor'].mean())

mmean.append(mb5560['valor'].mean())

mmean.append(mb6065['valor'].mean())

mmean.append(mb6570['valor'].mean())

mmean.append(mb7075['valor'].mean())

mmean.append(mb7580['valor'].mean())

mmean.append(mb8085['valor'].mean())

mmean.append(mb8590['valor'].mean())

mmean.append(ma90['valor'].mean())



mvaluestd = []

mvaluestd.append(mb50['valor'].std())

mvaluestd.append(mb5055['valor'].std())

mvaluestd.append(mb5560['valor'].std())

mvaluestd.append(mb6065['valor'].std())

mvaluestd.append(mb6570['valor'].std())

mvaluestd.append(mb7075['valor'].std())

mvaluestd.append(mb7580['valor'].std())

mvaluestd.append(mb8085['valor'].std())

mvaluestd.append(mb8590['valor'].std())

mvaluestd.append(ma90['valor'].std())



plt.subplot(2,2,3, title = 'Midfielders Market Value Means') 

plt.bar(labels, mmean, yerr = mvaluestd, color = colors)



ab50 = attack.loc[attack['Overall'] < 50]

ab5055 = attack.loc[(attack['Overall'] >=50) & (attack['Overall'] < 55)]

ab5560 = attack.loc[(attack['Overall'] >=55) & (attack['Overall'] < 60)]

ab6065 = attack.loc[(attack['Overall'] >=60) & (attack['Overall'] < 65)]

ab6570 = attack.loc[(attack['Overall'] >=65) & (attack['Overall'] < 70)]

ab7075 = attack.loc[(attack['Overall'] >=70) & (attack['Overall'] < 75)]

ab7580 = attack.loc[(attack['Overall'] >=75) & (attack['Overall'] < 80)]

ab8085 = attack.loc[(attack['Overall'] >=80) & (attack['Overall'] < 85)]

ab8590 = attack.loc[(attack['Overall'] >=85) & (attack['Overall'] < 90)]

aa90 = attack.loc[attack['Overall'] >= 90]



amean = []

amean.append(ab50['valor'].mean())

amean.append(ab5055['valor'].mean())

amean.append(ab5560['valor'].mean())

amean.append(ab6065['valor'].mean())

amean.append(ab6570['valor'].mean())

amean.append(ab7075['valor'].mean())

amean.append(ab7580['valor'].mean())

amean.append(ab8085['valor'].mean())

amean.append(ab8590['valor'].mean())

amean.append(aa90['valor'].mean())



avaluestd = []

avaluestd.append(ab50['valor'].std())

avaluestd.append(ab5055['valor'].std())

avaluestd.append(ab5560['valor'].std())

avaluestd.append(ab6065['valor'].std())

avaluestd.append(ab6570['valor'].std())

avaluestd.append(ab7075['valor'].std())

avaluestd.append(ab7580['valor'].std())

avaluestd.append(ab8085['valor'].std())

avaluestd.append(ab8590['valor'].std())

avaluestd.append(aa90['valor'].std())



plt.subplot(2,2,4, title = 'Attackers Market Value Means') 

plt.bar(labels, amean, yerr = avaluestd, color = colors)
def sumpartialoverall(s):

    s = str(s)

    if "+" in s:

        strp = s.split('+')

        onep = int(strp[0])

        twop = int(strp[1])

        final = onep + twop

    else:

        onep = int(s)

        final = onep

    return final
dataset['LS'] = dataset['LS'].apply(sumpartialoverall)

dataset['ST'] = dataset['ST'].apply(sumpartialoverall)

dataset['RS'] = dataset['RS'].apply(sumpartialoverall)

dataset['LW'] = dataset['LW'].apply(sumpartialoverall)

dataset['LF'] = dataset['LF'].apply(sumpartialoverall)

dataset['CF'] = dataset['CF'].apply(sumpartialoverall)

dataset['RF'] = dataset['RF'].apply(sumpartialoverall)

dataset['RW'] = dataset['RW'].apply(sumpartialoverall)

dataset['LAM'] = dataset['LAM'].apply(sumpartialoverall)

dataset['CAM'] = dataset['CAM'].apply(sumpartialoverall)

dataset['RAM'] = dataset['RAM'].apply(sumpartialoverall)

dataset['LM'] = dataset['LM'].apply(sumpartialoverall)

dataset['LCM'] = dataset['LCM'].apply(sumpartialoverall)

dataset['CM'] = dataset['CM'].apply(sumpartialoverall)

dataset['RCM'] = dataset['RCM'].apply(sumpartialoverall)

dataset['RM'] = dataset['RM'].apply(sumpartialoverall)

dataset['LWB'] = dataset['LWB'].apply(sumpartialoverall)

dataset['LDM'] = dataset['LDM'].apply(sumpartialoverall)

dataset['CDM'] = dataset['CDM'].apply(sumpartialoverall)

dataset['RDM'] = dataset['RDM'].apply(sumpartialoverall)

dataset['RWB'] = dataset['RWB'].apply(sumpartialoverall)

dataset['LB'] = dataset['LB'].apply(sumpartialoverall)

dataset['LCB'] = dataset['LCB'].apply(sumpartialoverall)

dataset['CB'] = dataset['CB'].apply(sumpartialoverall)

dataset['RCB'] = dataset['RCB'].apply(sumpartialoverall)

dataset['RB'] = dataset['RB'].apply(sumpartialoverall)

dataset['totaloverall'] = (dataset['LS'] + dataset['ST'] + dataset['RS'] + dataset['LW'] + dataset['LF']  + dataset['CF'] + dataset['RF'] + dataset['RW']

                          + dataset['LAM'] + dataset['CAM'] + dataset['RAM'] + dataset['LM'] + dataset['LCM'] + dataset['CM'] + dataset['RCM'] +

                            dataset['RM'] + dataset['LWB'] + dataset['LDM'] + dataset['CDM'] + dataset['RDM'] + dataset['RWB'] + dataset['LB'] +

                           dataset['LCB'] + dataset['CB'] + dataset['RCB'] + dataset['RB'])
dataset.head() #just to make sure its ok
copy = dataset

copy = dataset

copy.drop(copy[copy.totaloverall == 0].index, inplace = True)



copy['totaloverall'].hist(bins=100, color='red')
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,10)



copy.plot(x = 'totaloverall', y = 'valor', kind = 'scatter', title = 'Total Overall x Value', color='r')
%matplotlib inline



plt.rcParams['figure.figsize'] = (10,8)



rightfoot = dataset.loc[dataset['Preferred Foot'] == 'Right']

leftfoot = dataset.loc[dataset['Preferred Foot'] == 'Left']



overmeans = []

overmeans.append(rightfoot['Overall'].mean())

overmeans.append(leftfoot['Overall'].mean())



stdover = []

stdover.append(rightfoot['Overall'].std())

stdover.append(leftfoot['Overall'].std())



colors = ['#34ff00', '#ff9811' ]



lbs = ['Right foot', 'Left foot']



plt.subplot(1,2,1, title = 'Right/Left Foot Overall Means') 

plt.bar(lbs, overmeans, yerr = stdover, color = colors)



valuemeans = []

valuemeans.append(rightfoot['valor'].mean())

valuemeans.append(leftfoot['valor'].mean())



stdvalue = []

stdvalue.append(rightfoot['valor'].std())

stdvalue.append(leftfoot['valor'].std())



plt.subplot(1,2,2, title = 'Right/Left Foot Value Means') 

plt.bar(lbs, valuemeans, yerr = stdvalue, color = colors)
%matplotlib inline



plt.rcParams['figure.figsize'] = (10,8)



rightfoot = rightfoot.loc[(rightfoot['Overall'] >= 70) & (rightfoot['Overall'] <=80)]

leftfoot = leftfoot.loc[(leftfoot['Overall'] >= 70) & (leftfoot['Overall'] <=80)]



overmeans = []

overmeans.append(rightfoot['Overall'].mean())

overmeans.append(leftfoot['Overall'].mean())



stdover = []

stdover.append(rightfoot['Overall'].std())

stdover.append(leftfoot['Overall'].std())



colors = ['#34ff00', '#ff9811']



lbs = ['Right foot', 'Left foot']



plt.subplot(1,2,1, title = 'Right/Left Foot Overall Means') 

plt.bar(lbs, overmeans, yerr = stdover, color = colors)



valuemeans = []

valuemeans.append(rightfoot['valor'].mean())

valuemeans.append(leftfoot['valor'].mean())



stdvalue = []

stdvalue.append(rightfoot['valor'].std())

stdvalue.append(leftfoot['valor'].std())



plt.subplot(1,2,2, title = 'Right/Left Foot Value Means') 

plt.bar(lbs, valuemeans, yerr = stdvalue, color = colors)
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,8)



verylowweakfoot = dataset.loc[dataset['Weak Foot'] == 1]

lowweakfoot = dataset.loc[dataset['Weak Foot'] == 2]

mediumweakfoot = dataset.loc[dataset['Weak Foot'] == 3]

highweakfoot = dataset.loc[dataset['Weak Foot'] == 4]

veryhighweakfoot = dataset.loc[dataset['Weak Foot'] == 5]



om = []

om.append(verylowweakfoot['Overall'].mean())

om.append(lowweakfoot['Overall'].mean())

om.append(mediumweakfoot['Overall'].mean())

om.append(highweakfoot['Overall'].mean())

om.append(veryhighweakfoot['Overall'].mean())



ostd = []

ostd.append(verylowweakfoot['Overall'].std())

ostd.append(lowweakfoot['Overall'].std())

ostd.append(mediumweakfoot['Overall'].std())

ostd.append(highweakfoot['Overall'].std())

ostd.append(veryhighweakfoot['Overall'].std())



colors = ['#34ff00', '#ff9811', '#1209fe', '#561200', '#afeebc']



lbs = ['Very low weak foot', 'Low weak foot', 'Medium weak foot', 'High weak foot', 'Very high weak foot']



plt.subplot(1,2,1, title = 'Skills with bad foot Overall Means') 

plt.bar(lbs, om, yerr = ostd, color = colors)



vm = []

vm.append(verylowweakfoot['valor'].mean())

vm.append(lowweakfoot['valor'].mean())

vm.append(mediumweakfoot['valor'].mean())

vm.append(highweakfoot['valor'].mean())

vm.append(veryhighweakfoot['valor'].mean())



vstd = []

vstd.append(verylowweakfoot['valor'].std())

vstd.append(lowweakfoot['valor'].std())

vstd.append(mediumweakfoot['valor'].std())

vstd.append(highweakfoot['valor'].std())

vstd.append(veryhighweakfoot['valor'].std())



plt.subplot(1,2,2, title = 'Skills with bad foot Value Means') 

plt.bar(lbs, vm, yerr = vstd, color = colors)
%matplotlib inline



plt.rcParams['figure.figsize'] = (20,8)



verylowweakfoot = verylowweakfoot.loc[(verylowweakfoot['Overall'] >= 70) & (verylowweakfoot['Overall'] <=80)]

lowweakfoot = lowweakfoot.loc[(lowweakfoot['Overall'] >= 70) & (lowweakfoot['Overall'] <=80)]

mediumweakfoot = mediumweakfoot.loc[(mediumweakfoot['Overall'] >= 70) & (mediumweakfoot['Overall'] <=80)]

highweakfoot = highweakfoot.loc[(highweakfoot['Overall'] >= 70) & (highweakfoot['Overall'] <=80)]

veryhighweakfoot = veryhighweakfoot.loc[(veryhighweakfoot['Overall'] >= 70) & (veryhighweakfoot['Overall'] <=80)]



om = []

om.append(verylowweakfoot['Overall'].mean())

om.append(lowweakfoot['Overall'].mean())

om.append(mediumweakfoot['Overall'].mean())

om.append(highweakfoot['Overall'].mean())

om.append(veryhighweakfoot['Overall'].mean())



ostd = []

ostd.append(verylowweakfoot['Overall'].std())

ostd.append(lowweakfoot['Overall'].std())

ostd.append(mediumweakfoot['Overall'].std())

ostd.append(highweakfoot['Overall'].std())

ostd.append(veryhighweakfoot['Overall'].std())



colors = ['#34ff00', '#ff9811', '#1209fe', '#561200', '#afeebc']



lbs = ['Very low weak foot', 'Low weak foot', 'Medium weak foot', 'High weak foot', 'Very high weak foot']



plt.subplot(1,2,1, title = 'Skills with bad foot Overall Means') 

plt.bar(lbs, om, yerr = ostd, color = colors)



vm = []

vm.append(verylowweakfoot['valor'].mean())

vm.append(lowweakfoot['valor'].mean())

vm.append(mediumweakfoot['valor'].mean())

vm.append(highweakfoot['valor'].mean())

vm.append(veryhighweakfoot['valor'].mean())



vstd = []

vstd.append(verylowweakfoot['valor'].std())

vstd.append(lowweakfoot['valor'].std())

vstd.append(mediumweakfoot['valor'].std())

vstd.append(highweakfoot['valor'].std())

vstd.append(veryhighweakfoot['valor'].std())



plt.subplot(1,2,2, title = 'Skills with bad foot Value Means') 

plt.bar(lbs, vm, yerr = vstd, color = colors)