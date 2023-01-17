import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
metaData1 = pd.read_csv("../input/csgo-matchmaking-damage/esea_meta_demos.part1.csv")

metaData2 = pd.read_csv("../input/csgo-matchmaking-damage/esea_meta_demos.part2.csv")

metaData = pd.concat([metaData1, metaData2]) # combine the data which was seperate
metaData.head() # so you can see what the data looks like

# ignore the winner_team as the majority of the rounds it is simply Team 1 or Team 2
metaData.shape
# picks out the relavent data and groups them by game and map

mapPrevalence = metaData.groupby(['file','map']).count().reset_index()[['file','map']]

# renames the column and counts the number of times a map is seen

mapPrevalence = mapPrevalence['map'].value_counts().reset_index().rename(columns = {'index':'Map', 'map':'Times Played'})



# plots the data

fig, ax = plt.subplots(figsize = (12,8))

sns.barplot(x='Map', y='Times Played', data = mapPrevalence)
mapPrevalence
# finding the number of rounds in each game and renaming the columns

gameRounds = metaData[['file','round']].groupby('file').max().rename(columns = {'round':'Number of Rounds'}).reset_index()
# some of the data for the games do not begin at round 1 as can be seen below

metaData[['file','round']].groupby('file').min()['round'].value_counts()
# it was going to be a little tedious to filter the start time of each round because of the above. So I just found an average

# start time and used that. The average game times are 2268 second which is small compared to the std of the start times.

startTime = metaData[['file','start_seconds']].groupby('file').min().mean()

startTimeStd = metaData[['file','start_seconds']].groupby('file').min().std()

print(startTime)

print(startTimeStd)
# finding the total game time

gameTime = metaData[['file','end_seconds']].groupby('file').max()

# renaming the columns and subtracting the average start time

gameTime = gameTime.rename(columns = {'end_seconds':'Game Time /min'})-startTime.iloc[0]

print(gameTime.mean())

# convert from seconds to minutes

gameTime /= 60
# plot the results for the number of rounds and minutes per game



fig, ax = plt.subplots(figsize=(18,4))

sns.violinplot(x = 'Number of Rounds', data = gameRounds, ax = ax)



fig2, ax2 = plt.subplots(figsize=(18,4))

sns.violinplot(x = 'Game Time /min', data = gameTime, ax = ax2)
# create a column with the round duration

metaData['Round Duration /s'] = metaData['end_seconds']-metaData['start_seconds']

# remove rounds that lasted more than 180 seconds as they must be erroneous

duration = metaData[metaData['Round Duration /s'] < 180]

#calculate the mean and std of the duration and present the data

durationMean = duration[['map','Round Duration /s']].groupby('map').mean()

durationStd = duration[['map','Round Duration /s']].groupby('map').std().rename(columns = {'Round Duration /s':'Sigma /s'})

pd.concat([durationMean, durationStd], axis = 1)
# plotting the data

fig, ax1 = plt.subplots(figsize = (8, 12))

sns.set(font_scale = 1.3)

sns.violinplot(y = 'map', x = 'Round Duration /s', data = duration)
# extracting the relavent columns and renaming them

metaDataFormatted = metaData[['round_type','Round Duration /s']].rename(columns = {'round_type':'Round Type'})

# renaming the round types so they are more presentable

metaDataFormatted['Round Type'] = metaDataFormatted['Round Type'].map({'PISTOL_ROUND':'Pistol Round','ECO':'Eco','SEMI_ECO':'Semi-eco',

                                                         'NORMAL':'Normal','FORCE_BUY':'Force Buy'})



# plotting the data

fig, ax = plt.subplots(figsize=(12,8))

sns.violinplot(y='Round Type', x='Round Duration /s', data=metaDataFormatted, ax=ax)

ax.set_xlim(0,200)
# define an empty data frame to hold the data about who won the rounds

wins = pd.DataFrame()

# extract a dataframe with the relavent information

winner = metaData[['map','winner_side']]

for mapName in metaData['map'].unique(): # loop through the different maps that are played

    # calculate the number of times Ts and CTs one per map type

    tWins = winner[(winner['map']==mapName) & (winner['winner_side']=='Terrorist')].shape[0]

    ctWins = winner[(winner['map']==mapName) & (winner['winner_side']=='CounterTerrorist')].shape[0]

    # calculate it as a percentage and append all the data to the wins dataframe

    total = tWins + ctWins

    dataToAdd = pd.DataFrame({'Map':[mapName, mapName], 'Wins':[ctWins, tWins], 'Winning Side':['CT','T'], 

                              'Winning Percentage':[ctWins/total*100, tWins/total*100],

                               'Sidedness':[(ctWins-tWins)/total*100, (tWins-ctWins)/total*100]})

    wins = wins.append(dataToAdd, ignore_index = True)

wins
# plot the data from above



fig1, ax1 = plt.subplots(figsize=(12,8))

sns.set(font_scale = 1.3)

sns.barplot(x='Map',y='Winning Percentage',hue='Winning Side',data=wins, ax= ax1)

ax1.legend(loc = 7)



fig2, ax2 = plt.subplots(figsize=(12,8))

filteredWins = wins[wins['Winning Side'] == 'T'][['Map','Sidedness']].rename(columns={'Sidedness':'T-Sidedness'})

sns.barplot(x='Map', y = 'T-Sidedness', palette = ['sandybrown'], data = filteredWins, ax = ax2)
killsData1 = pd.read_csv("../input/csgo-matchmaking-damage/esea_master_kills_demos.part1.csv")

killsData2 = pd.read_csv("../input/csgo-matchmaking-damage/esea_master_kills_demos.part2.csv")

killsData = pd.concat([killsData1, killsData2])
killsData.head()
# extract the data related to the weapons used on each kill

weaponData = killsData.groupby(['wp_type','wp']).count().reset_index()[['wp_type','wp','file']]

weaponData = weaponData.rename(columns = {'wp_type':'Weapon Type', 'wp':'Weapon', 'file':'Kills'})



# create a subplot for each weapon type

fig, axs = plt.subplots(ncols=1, nrows=8, figsize = (15,25)) # create 8 subplots

for ax_num in range(len(axs)): # loop through the subplots and plot the relavent data

    weaponType = weaponData['Weapon Type'].unique()[ax_num]

    dataToPlot = weaponData[weaponData['Weapon Type']==weaponType]

    sns.barplot(x='Weapon', y='Kills', data=dataToPlot, ax = axs[ax_num])
# extracting all the raw columns that don't need additional processing

dataProcessing1 = metaData[['round','ct_eq_val','t_eq_val']]



# converting the map data into a binary input for each map so it can be processed by the machine

maps = metaData['map'].unique() # order: overpass, cache, inferno, muirage, train, dust2, cobble, nuke

mapsBin = pd.DataFrame()

for mapp in maps:

    mapsBin[mapp] = metaData['map'].apply(lambda x: 1 if x == mapp else 0)

    

# creating the binary column with who won the previous round

# first create a series with a single value (to shift the series down)

s = pd.Series([0])

# creating a series with 1 if the it is not the first round and 0 if it is

isNotFirstRound = metaData['round'].apply(lambda x: 0 if x == 1 else 1).rename('isNotFirstRound').reset_index(drop=True)

# append the series from above, shifting the series down (now when the series are put together match up with the previous

#   match)

shiftedWinnerSide = s.append(metaData['winner_side'])

# create the two series with the binary output whether Ts or CTs won (cutting off the last row as it not meaningful)

tWinPrev = shiftedWinnerSide.apply(lambda x: 1 if x=='Terrorist' else 0).iloc[:-1]

ctWinPrev = shiftedWinnerSide.apply(lambda x: 1 if x=='CounterTerrorist' else 0).iloc[:-1]

# resetting the index for the next part

tWinPrev = tWinPrev.reset_index(drop=True)

ctWinPrev = ctWinPrev.reset_index(drop=True)

# multiply each series by the "isNotFirstRound" from above. This ensures that on pistol round nobody has won previously,

#    without this first rounds would have a previous winner and the machine may attempt to find a pattern in that

tWinPrev = (tWinPrev * isNotFirstRound).rename('tWinPrev')

ctWinPrev = (ctWinPrev * isNotFirstRound).rename('tWinPrev')

# combine the two series into a single dataframe

prevWinDF = pd.concat([tWinPrev,ctWinPrev], axis = 1)



# convert the round winner to a binary number. This is the data the machine will attempt to predict

dataToPredict = metaData['winner_side'].apply(lambda x: 1 if x=='Terrorist' else 0)
# creating a list of the different data categories we will use, you can add and remove these as you please

processedCols = [dataProcessing1, mapsBin, tWinPrev, ctWinPrev, dataToPredict]

# concatenate them into a single dataframe

processedData = pd.concat(processedCols, axis = 'columns', join = 'inner')

# take a peek at the feature list

print('Feature list:')

processedData.columns
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, f1_score, confusion_matrix
# separtate the data into X and y sets. The datasets are shortened to cut down on processing time! Feel free to crank the set size up or down depending

# on your resources. You can change this and then run an individual classifier.

subset_size = 5000

X = processedData.drop('winner_side', axis = 1)[:subset_size]

y = processedData['winner_side'][:subset_size]

# split them into testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)



# scale the data

ss = StandardScaler()

X_train = ss.fit_transform(X_train)

X_test = ss.transform(X_test)
# instantiate lists to hold the values

Ks = []

Kscores = []

# loop through different K values, testing each one, to find the optimal one

for K in range(15,100):

    kClassifier = KNeighborsClassifier(n_neighbors=K)

    kClassifier.fit(X_train, y_train)

    y_pred = kClassifier.predict(X_test)

    Kscores.append(f1_score(y_test, y_pred))

    Ks.append(K)



# extract and print the best K value

bestK = Ks[Kscores.index(max(Kscores))]

print(f'Optimal K-value: {bestK}')



kClassifier = KNeighborsClassifier(n_neighbors=bestK)

kClassifier.fit(X_train, y_train)

y_pred = kClassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.svm import SVC



# instantiate hyper variable and scores lists

gammas = []

Cs = []

svmScores = []



# loop through the data, changing gamma and C to find an optimal value. This takes a while depending on the number of cases you test so be careful.

for C in np.linspace(0, 10, 20):

    for gamma in np.linspace(0.01, 1, 20):

        SVM = SVC(C = C, gamma = gamma)

        SVM.fit(X_train, y_train)

        score = SVM.score(X_test, y_test)

        svmScores.append(score)

        gammas.append(gamma)

        Cs.append(C)

        

# extract the best C and gamma values

index = svmScores.index(max(svmScores))

bestC = Cs[index]

bestGamma = gammas[index]



print(f'Best C value: {bestC}')

print(f'Best gamma value: {bestGamma}')



SVM = SVC(C=bestC, gamma=bestGamma)

SVM.fit(X_train, y_train)

y_pred = SVM.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
depths = []

RFCscores = []

for depth in range(2, 16):

    forest = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42)

    forest.fit(X_train, y_train)

    y_pred = forest.predict(X_test)

    depths.append(depth)

    RFCscores.append(f1_score(y_test, y_pred))

    #print(confusion_matrix(y_test, y_pred))

    #print(classification_report(y_test, y_pred))

    

bestDepth = depths[RFCscores.index(max(RFCscores))]

print(f'Best depth setting: {bestDepth}')



forest = RandomForestClassifier(n_estimators=500, max_depth=bestDepth, random_state=42)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
# feel free to mess around with all the values here to try and get better scores, it's pretty barebones right now.

MLPC = MLPClassifier(hidden_layer_sizes=(11,11), activation='relu',

                     batch_size=500, max_iter=200, random_state=42)

MLPC.fit(X_train, y_train)

y_pred = MLPC.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
# creates a table indicating whether the bomb was planted for each round. 

# requires somebody to die after the bomb was planted to work and so is not entirely accurate !

killsData['binBombPlant'] = killsData['is_bomb_planted'].map({False:0, True:1})

bombPlantData = killsData[['file','round','binBombPlant']].groupby(['file','round']).max().reset_index()

bombPlantData.head()
# gives how many of each team remained alive after each round

killsData.groupby(['file','round'])[['ct_alive', 't_alive']].min()