%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')
lol = pd.read_csv('../input/LeagueofLegends.csv')



lol = lol.drop('Address', axis = 1) 

#The match history is not very useful, for now, in this analysis.

lol.head()
lol.describe()#Here we do a little statics describe
lol['gamelength'].value_counts().sort_index().plot()
lol['gamelength'].plot.hist()
lol[lol['gamelength']>80]
lol[lol['gamelength']<18]
lol.dtypes #here we see that we'll not have problems with the types
lol2016 = lol[lol['Year'] >= 2016] 

#2016 It's when the CBLOL starts to get recorde on the dataset.



#Here I make the graphs of each region.

lcsna = lol2016[lol2016['League'] == 'NALCS']



lcseu = lol2016[lol2016['League'] == 'EULCS']



lck = lol2016[lol2016['League'] == 'LCK']



lms = lol2016[lol2016['League'] == 'LMS']



cblol = lol2016[lol2016['League'] == 'CBLoL']
top = 5 #Here it's a variable so I can change How many teams of each region I want to see.



leagues = {

'CBLOL': (cblol['blueTeamTag'].value_counts() + cblol['redTeamTag'].value_counts()).sort_values(ascending = False).head(top),

'LCSNA': (lcsna['blueTeamTag'].value_counts() + lcsna['redTeamTag'].value_counts()).sort_values(ascending = False).head(top),

'LCSEU': (lcseu['blueTeamTag'].value_counts() + lcseu['redTeamTag'].value_counts()).sort_values(ascending = False).head(top),

'LCK': (lck['blueTeamTag'].value_counts() + lck['redTeamTag'].value_counts()).sort_values(ascending = False).head(top),

'LMS': (lms['blueTeamTag'].value_counts() + lms['redTeamTag'].value_counts()).sort_values(ascending = False).head(top),

}



allGames = pd.DataFrame()



total = pd.Series()

total.name = 'total'



for i in leagues:

    leagues[i].name=i

    allGames = allGames.append(leagues[i]).fillna(0)

    total = total.append(leagues[i])

    

allGames = allGames.transpose()



allGames['total'] = total

allGames = allGames.sort_values('total')

allGames = allGames.drop('total', axis = 1)



allGames.plot.barh(figsize = (10,10), grid = False, stacked = True)



plt.title('Top %s Teams Who Played The Most Games In Their Regions (2016-2017)' % top)

plt.ylabel('Teams')

plt.xlabel('Matches')

plt.legend()
champions = lol2016.loc[:, 'blueTopChamp':'redSupportChamp']

aux = 0

for i in champions:

    

    if 'Champ' not in i:

        champions = champions.drop(i, axis = 1) 

    else:

        aux += champions[i].value_counts()



a = pd.Series()

for i in champions:

    a = (a+champions[i].value_counts().sort_values(ascending=False)).fillna(0)

    

a[:] = 0

champs = a.copy()

for i in champions:

    champs = champs+(a+champions[i].value_counts().sort_values(ascending=False)).fillna(0)

    

print(champs.sort_values(ascending = False).head(10))

"""

for i in champions:

    if 'Gragas' in champions[i].value_counts():

        print(champions[i].value_counts()['Gragas'])

"""

champs = champs.sort_values(ascending = False)

champs.values.sum()
champs.head(10).plot(kind = 'bar', figsize=(15,6), grid = False, rot = 0, color = 'red')



plt.title('Top 10 Champions Most Picked In Leagues')

plt.ylabel('Matches')

plt.xlabel('Champion')
champions = cblol.loc[:, 'blueTopChamp':'redSupportChamp']



for i in champions:

    

    if 'Champ' not in i:

        champions = champions.drop(i, axis = 1) 

    else:

        aux += champions[i].value_counts()



a = 0

for i in champions:

    a=a+champions[i].value_counts().sort_values(ascending=False)



a = a.fillna(0)

championsCblol = 0

for i in champions:

    championsCblol+=(a+champions[i].value_counts().sort_values(ascending=False)).fillna(0)



print(championsCblol.sort_values(ascending = False).head(10))

championsCblol = championsCblol.sort_values(ascending = False)



championsCblol.head(10).plot(kind = 'bar', figsize=(15,6), grid = False, rot = 0, color = 'green')



plt.title('Top 10 Champions Most Used In CBLOL')

plt.ylabel('Matches')

plt.xlabel('Champion')
shape = 0

for i in champions:

    if 'blue' in i:

        shape = shape + pd.crosstab( lol2016[i] ,lol2016['bResult']).fillna(0)

    if 'red' in i:

        shape = shape + pd.crosstab( lol2016[i] ,lol2016['rResult']).fillna(0)

        

shape[:] = 0



victories = 0

defeats = 0



for i in champions:

    if 'blue' in i:

        victories += (shape + pd.crosstab( lol2016[i] ,lol2016['bResult'])).fillna(0)[1]

        defeats += (shape + pd.crosstab( lol2016[i] ,lol2016['bResult'])).fillna(0)[0]

    if 'red' in i:

        victories += (shape + pd.crosstab( lol2016[i] ,lol2016['rResult'])).fillna(0)[1]

        defeats += (shape + pd.crosstab( lol2016[i] ,lol2016['rResult'])).fillna(0)[0]



#print(victories)

        

X = pd.DataFrame()

X['total'] = champs.copy()

X['victories'] = victories.copy()

X['defeats'] = defeats.copy()

X['winRate'] = X['victories']/X['total']



X = X[X['total']>=100].sort_values('winRate', ascending = False)



plt.figure(figsize=(5, 5))



X['winRate'].head(10).sort_values().plot.barh() 

plt.title('Top 10 Champions Win Rate (More Than 100 games) in World 2016-2017')

plt.xlabel('Win Rate')

plt.ylabel('Champion')

plt.show()

shape = pd.DataFrame()

for i in champions:

    shape = shape + pd.crosstab( lol2016[i] ,lol2016['League']).fillna(0)



shape[:] = 0

shape = shape.drop(['IEM', 'MSI', 'WC'], axis = 1)

champions_Leagues = shape.copy()



for j in champions:

    champions_Leagues += (shape + pd.crosstab( lol2016[j] ,lol2016['League'])).fillna(0)



    

total = champions_Leagues['CBLoL'].copy()

total[:]=0

for i in champions_Leagues:

    total = (total + champions_Leagues[i]).fillna(0)



champions_Leagues['total'] = total



sns.set(font_scale=1.2)

plt.figure(figsize=(15, 8))

champions_Leagues = champions_Leagues.sort_values(by = 'total', ascending = False).head(10)

sns.heatmap(champions_Leagues, annot=True, vmax=champions_Leagues.loc[:, 'CBLoL':'TCL'].values.max(), vmin=champions_Leagues.loc[:, 'CBLoL':'TCL'].values.min(), fmt='g') 



plt.title('10 Most Picked Champions Heat Map By Region')

plt.xlabel('Region')

plt.ylabel('Champion')

plt.show()

total.idxmax()
