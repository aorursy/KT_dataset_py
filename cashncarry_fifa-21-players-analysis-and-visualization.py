import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
# import datasets

df = pd.read_csv('../input/fifa-21-players-teams-full-database/players_fifa21.csv')

df.head()
df.isna().sum().sort_values(ascending=False)[:10]
promising_players = df[(df.Growth > 4) & (df.Potential > 84)].sort_values(by='Potential', ascending=False)

print(f'Players find: {promising_players.shape[0]}')
promising_players[

    ['Name', 'Age', 'Overall', 'Potential', 'BestPosition', 'ValueEUR', 'ReleaseClause']

].head(20).style.background_gradient(cmap='cool')
promising_players[

    ['Name', 'Age', 'Overall', 'Potential', 'BestPosition', 'ValueEUR', 'ReleaseClause']

].corr().style.background_gradient(cmap='cool')
plt.figure(figsize=(12,6))

sns.boxplot(x='BestPosition', 

            y='Potential', 

            data = promising_players, 

            palette='cool'

           ).set_title('Potential players by position');
populat_nationality = promising_players.Nationality.value_counts()[:10].keys()

plt.figure(figsize=(12,6))

sns.boxplot(x='Nationality', 

            y='Potential', 

            data = promising_players[promising_players.Nationality.isin(populat_nationality)], 

            palette='cool'

           ).set_title('Potential players by nationality');
plt.figure(figsize=(16,6))

plt.xticks(rotation=340)

club_count_players = promising_players.Club.value_counts()[:20]

sns.barplot(x=club_count_players.index, 

            y=club_count_players.values,

            palette='Set2'

           ).set_title('Talents in clubs');
from IPython.display import display, HTML

printing_cols =['Name', 'Age', 'Overall', 'Potential', 'Positions', 'ValueEUR', 'ReleaseClause']

for pos in promising_players.BestPosition.unique():

    print(f'Best {pos}')

    display(HTML(promising_players[promising_players.BestPosition == pos][printing_cols].head(5).to_html()))

    print('\n\n')
print("FIFA is far from real football...")

df.iloc[:, -17:].corr().style.background_gradient(cmap='cool')
from math import pi



def make_spider(row, i):

    # Create Radar chart

    categories=['PassingTotal', 'ShootingTotal', 'PaceTotal', 

                'PhysicalityTotal',  'DefendingTotal', 'DribblingTotal']

    if row[1].BestPosition == "GK":

        categories=['Reactions', 'GKDiving', 'GKHandling', 

                    'GKKicking', 'GKPositioning', 'GKReflexes']

    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]

    ax = plt.subplot(1, 3, i + 1, polar=True)

    plt.subplots_adjust(wspace=0.6)

    xticks_names =  [f"{cat.replace('Total', '')}\n{row[1][cat]}" for cat in categories]

    plt.xticks(angles[:-1], xticks_names, color='grey', size=10)

 

    ax.set_rlabel_position(0)

    ax.set_facecolor('xkcd:black')

    plt.yticks([40, 50, 60, 70, 80, 90], 

               ["40", "50", "60","70","80","90"], 

               color="grey", size=8)

    plt.ylim(0,100)

 



    values=row[1][categories].values.flatten().tolist()

    values += values[:1]

    ax.plot(angles, values, color='cyan', linewidth=3, linestyle='solid')

    ax.fill(angles, values, color='cyan', alpha=0.6)

 

    plt.title(row[1].Name, size=14, color='black', y=1.1)



positions = ['ST', 'CF', 'RF', 'LF', 'RW', 'LW', 'LM', 'RM', 

             'CAM', 'CM', 'CDM', 'RB', 'LB', 'CB', 'RWB', 'LWB', 'GK']

bar_kwargs = dict(x='Name', y='Overall', figsize=(16,4), yticks=list(range(0, 100, 10)), rot=0)



for pos in positions:

    print(f'\n\nBest in {pos}')

    df[df.ClubPosition==pos][:10].plot.bar(**bar_kwargs);

    plt.show()

    plt.figure(figsize=(16,16))

    for i, row in enumerate(df[df.ClubPosition==pos][:3].iterrows()):

        make_spider(row, i)
df[df.ContractUntil < 2022][

    ['Name', 'Age', 'Overall', 'Potential', 'Positions', 'ContractUntil', 'WageEUR']

][:10].style.background_gradient(cmap='cool')
print('Best under 27')

df[(df.ContractUntil < 2022) & (df.Age < 27)][

    ['Name', 'Age', 'Overall', 'Potential', 'Positions', 'ContractUntil', 'WageEUR']

][:20].style.background_gradient(cmap='cool')
print('Young and talented')

df[(df.ContractUntil < 2022) & (df.Age < 25) & (df.Growth > 4)][

    ['Name', 'Age', 'Overall', 'Potential', 'Positions', 'ContractUntil', 'WageEUR']

][:20].style.background_gradient(cmap='cool')
def highlight_low(s):

    if s.dtype == 'O':

        return s

    is_low = s <= s.quantile(0.15)

    return ['background-color: hotpink' if v else '' for v in is_low]





rating_columns = ['Name', 'Positions'] + list(df.columns[33:39])
df[df.BestPosition == 'CM'][:10][rating_columns].style.apply(highlight_low)
df[df.BestPosition == 'ST'][:10][rating_columns].style.apply(highlight_low)
df[df.BestPosition == 'CB'][:10][rating_columns].style.apply(highlight_low)