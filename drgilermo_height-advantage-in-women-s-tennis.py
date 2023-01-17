import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import seaborn as sns

plt.style.use('fivethirtyeight')
path = "../input/"

os.chdir(path)

filenames = os.listdir(path)

df = pd.DataFrame()

for filename in sorted(filenames):

    try:

        read_filename = '../input/' + filename

        temp = pd.read_csv(read_filename,encoding='utf8')

        frame = [df,temp]

        df = pd.concat(frame)

    except UnicodeDecodeError:

        pass
df['Year'] = df.tourney_date.apply(lambda x: str(x)[0:4])

df['Sets'] = df.score.apply(lambda x: x.count('-'))

df['Rank_Diff'] =  df['loser_rank'] - df['winner_rank']

df['Rank_Diff_Round'] = df.Rank_Diff.apply(lambda x: 10*round(np.true_divide(x,10)))

df['ind'] = range(len(df))

df = df.set_index('ind')
df['Height_Diff'] = df.winner_ht - df.loser_ht

sns.kdeplot(df.Height_Diff)

plt.xlim([-25,25])

plt.xlabel(['Height Difference'])
winners = list(np.unique(df.winner_name))

losers = list(np.unique(df.loser_name))



all_players = winners + losers

players = np.unique(all_players)



players_df = pd.DataFrame()

players_df['Name'] = players

players_df['Wins'] = players_df.Name.apply(lambda x: len(df[df.winner_name == x]))

players_df['Losses'] = players_df.Name.apply(lambda x: len(df[df.loser_name == x]))



players_df['PCT'] = np.true_divide(players_df.Wins,players_df.Wins + players_df.Losses)

players_df['Games'] = players_df.Wins + players_df.Losses

#%%





surfaces = ['Hard','Grass','Clay','Carpet']

for surface in surfaces:

    players_df[surface + '_wins'] = players_df.Name.apply(lambda x: len(df[(df.winner_name == x) & (df.surface == surface)]))

    players_df[surface + '_losses'] = players_df.Name.apply(lambda x: len(df[(df.loser_name == x) & (df.surface == surface)]))

    players_df[surface + 'PCT'] = np.true_divide(players_df[surface + '_wins'],players_df[surface + '_losses'] + players_df[surface + '_wins'])

    

serious_players = players_df[players_df.Games>40]

serious_players['Height'] = serious_players.Name.apply(lambda x: list(df.winner_ht[df.winner_name == x])[0])

serious_players['Best_Rank'] = serious_players.Name.apply(lambda x: min(df.winner_rank[df.winner_name == x]))

serious_players['Win_Aces'] = serious_players.Name.apply(lambda x: np.mean(df.w_ace[df.winner_name == x]))

serious_players['Lose_Aces'] = serious_players.Name.apply(lambda x: np.mean(df.l_ace[df.loser_name == x]))

serious_players['Aces'] = (serious_players['Win_Aces']*serious_players['Wins'] + serious_players['Lose_Aces']*serious_players['Losses'])/serious_players['Games']

sns.kdeplot(serious_players.Height)

plt.xlabel('Height')
plt.figure()

plt.plot(serious_players.Height + np.random.normal(0,1,len(serious_players)),serious_players.Aces,'o', markersize = 10, alpha = 0.7)

plt.xlabel('Height [cm]')

plt.ylabel('Aces Per Match')

plt.title('Height Matters #1 - More Aces')

x = serious_players.Height[(np.isnan(serious_players.Height) == 0) &  (np.isnan(serious_players.Aces) == 0 )]

y = serious_players.Aces[(np.isnan(serious_players.Height) == 0) &  (np.isnan(serious_players.Aces) == 0 )]

fit = np.polyfit(x,y,1)

fit_fn = np.poly1d(fit)

plt.plot(x,fit_fn(x))



plt.figure()

plt.plot(serious_players.Height + np.random.normal(0,1,len(serious_players)),serious_players.Best_Rank+ np.random.normal(0,1,len(serious_players)) ,'o', markersize = 10, alpha = 0.7)

plt.xlabel('Height [cm]')

plt.ylabel('Highest Career Rank')

plt.title('Height Matters #2 - Best Career Rank')

x = serious_players.Height[(np.isnan(serious_players.Height) == 0) &  (np.isnan(serious_players.Best_Rank) == 0 )]

y = serious_players.Best_Rank[(np.isnan(serious_players.Height) == 0) &  (np.isnan(serious_players.Best_Rank) == 0 )]

fit = np.polyfit(x,y,1)

fit_fn = np.poly1d(fit)

plt.plot(x,fit_fn(x))



plt.figure()

x = serious_players.Height[(np.isnan(serious_players.Height) == 0) & (np.isnan(serious_players.PCT) == 0 )]

y = serious_players.PCT[(np.isnan(serious_players.Height) == 0) & (np.isnan(serious_players.PCT) == 0 )]

plt.plot(x + np.random.normal(0,1,len(x)),y,'o', markersize = 10, alpha = 0.7)

plt.xlabel('Height [cm]')

plt.ylabel('Aces Per Match')

plt.title('Height Matters #3 - General PCT')



fit = np.polyfit(x,y,1)

fit_fn = np.poly1d(fit)

plt.plot(x,fit_fn(x))
print('Average Player Height', np.mean(serious_players.Height))

print('Average Height of Rank Leaders',np.mean(serious_players.Height[serious_players.Best_Rank == 1]))
height_diff_df = pd.DataFrame()

height_diff_df['height_bins'] = np.unique(df.Height_Diff)

height_diff_df['Height_Prob'] = height_diff_df.height_bins.apply(lambda x: np.true_divide(len(df[df.Height_Diff == x]),(len(df[df.Height_Diff == x]) +len(df[df.Height_Diff == -x]))))



x = height_diff_df.height_bins[30:55]

y = height_diff_df.Height_Prob[30:55]

plt.plot(x,y,'o', markersize = 10)

fit = np.polyfit(x,y,1)

fit_fn = np.poly1d(fit)

plt.plot(x,fit_fn(x))

plt.xlim([0,25])

plt.ylim([0.5,0.7])

plt.xlabel('Height Difference')

plt.ylabel('Winning Chanches')
avg_height = []

years = np.arange(2000,2017)

for year in years:

    avg_winner = np.mean(df.winner_ht[df.Year == str(year)])

    avg_loser = np.mean(df.winner_ht[df.Year == str(year)])

    avg_height.append(np.mean([avg_winner,avg_loser]))



plt.bar(years,avg_height)

plt.ylim([165,175])

plt.xlabel('Year')

plt.ylabel('Average Height')

plt.title('Are tennis players getting taller?')