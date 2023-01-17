import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import matplotlib

from plotly.offline import iplot

from plotly import figure_factory as FF



%matplotlib inline



plt.style.use('ggplot')

matplotlib.rcParams['font.family'] = "serif"

#sns.set_context('poster')





all_matches = pd.read_csv('/kaggle/input/cricket-odi-results/ODIMatchResults_Inn.csv')

all_matches['Date'] = pd.to_datetime(all_matches.Date, format="%d %b %Y")

all_matches['year'] = all_matches.Date.map(lambda x:x.year)
def function(x):

    matches = all_matches[(all_matches.Inn == 1) & (all_matches.runs >= x)]

    matches.year.value_counts(sort=False).plot.line(figsize=(20,6))

    plt.xlabel('Year', fontsize=25)

    string = str(x)

    plt.ylabel('Number of ' + string + '+ Scores', fontsize=20)

    ax = plt.gca()

    ax.xaxis.set_label_coords(0.5, -0.15)

    ax.yaxis.set_label_coords(-0.06, 0.5)

    ax.tick_params(labelsize=15)

    plt.show()

    

    total_matches = matches.Inn.count()

    total_matches_won = matches[matches.result == 'won'].Inn.count()



    table_data = [[string + '+ Batting First ( All Teams )', '# Won', '# Loss', 'Percent Win'],

                  [total_matches,total_matches_won,total_matches-total_matches_won,round((total_matches_won/total_matches)*100,2)]]

    figure1 = FF.create_table(table_data)



    teamWise_matches = matches.groupby('team1').Inn.count().sort_values(ascending=False)

    teamWise_matchesWon = matches[matches.result == 'won'].groupby('team1').Inn.count()

    teamWise_matchesWon = teamWise_matchesWon.reindex(teamWise_matches.index)

    table_data2 = []

    for i in range(0,len(teamWise_matches)):

        row = {}

        row['Team'] = teamWise_matches.index[i]

        row['# ' + string + '+ Batting First'] = teamWise_matches[i]

        row['# Won'] = teamWise_matchesWon[i]

        row['Percent Won'] = round((teamWise_matchesWon[i]/teamWise_matches[i])*100,2)

        table_data2.append(row)



    table_data2 = pd.DataFrame(table_data2)

    table_data2.sort_values(by='Percent Won',ascending=False, inplace=True)

    figure2 = FF.create_table(table_data2)

    return figure1, figure2



figure1, figure2 = function(300)
iplot(figure1)
iplot(figure2)
years = np.arange(1985,2020)

perwon = []

matches = all_matches[(all_matches.Inn == 1) & (all_matches.runs >= 300)]

for i in years:

    b = matches[matches.year <= i].result.value_counts()

    if(len(b) == 1):

        perwon.append(100)

    else:

        perwon.append((b[0]/(b[0]+b[1]))*100)

fig = plt.figure(figsize=(20,6))

ax = fig.add_subplot(111)

plt.plot(years, perwon)

plt.xlabel('Year', fontsize=25)

plt.ylabel('Win %', fontsize=20)

ax.xaxis.set_label_coords(0.5, -0.15)

ax.yaxis.set_label_coords(-0.05, 0.5)

ax.tick_params(labelsize=15)

plt.show()
figure1 , figure2 = function(325)

iplot(figure1)

iplot(figure2)
figure1 , figure2 = function(350)

iplot(figure1)

iplot(figure2)
matches = all_matches[(all_matches.Inn == 2) & (all_matches.runs>=300)]



total_matches = matches.groupby('team1').Inn.count()

total_matches_won = matches[matches.result == 'won'].groupby('team1').Inn.count()

total_matches_won = total_matches_won.reindex(total_matches.index)

table_data = []

total_matches_won[0] = 0

for i in range(0,len(total_matches_won)):

    row = {}

    row['Team'] = total_matches_won.index[i]

    row['Target of 300+'] = total_matches[i]

    if(total_matches_won[i] == None):

        total_matches_won[i] = 0

    row['# Successfully chased'] = int(total_matches_won[i])

    row['Win Percent'] = round((total_matches_won[i]/total_matches[i])*100,2)

    table_data.append(row)



table_data = pd.DataFrame(table_data)

table_data.sort_values(by='Win Percent',ascending=False, inplace=True)

figure = FF.create_table(table_data)

iplot(figure)