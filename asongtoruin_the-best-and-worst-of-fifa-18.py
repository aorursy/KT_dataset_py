import re

from collections import OrderedDict



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('darkgrid')
all_data = pd.read_csv('../input/CompleteDataset.csv', low_memory=False)

all_data.columns
team_avg = all_data.groupby('Club').agg({'Overall': 'mean',

                                         'Name': 'count'}).reset_index()

team_avg.sort_values(by='Overall', ascending=False, inplace=True)

team_avg.rename(columns={'Name': 'Squad Size'}, inplace=True)

team_avg
top_teams = team_avg.nlargest(n=20, columns=['Overall'])['Club'].tolist()

bottom_teams = team_avg.nsmallest(n=20, columns=['Overall'])['Club'].tolist()
fix, ax = plt.subplots(figsize=[10, 8])

top_teams_players = all_data[all_data['Club'].isin(top_teams)].copy()

top_teams_players['Club'] = top_teams_players['Club'].astype("category")

top_teams_players['Club'].cat.set_categories(top_teams, inplace=True)



sns.boxplot(data=top_teams_players, x='Overall', y='Club', showmeans=True)

ax.set_xlim([45, 100])
fix, ax = plt.subplots(figsize=[10, 12], sharey=True)

sns.violinplot(data=top_teams_players, x='Overall', y='Club',

               inner='quartile', bw=.2, ax=ax)

ax.set_xlim([45, 100])
fix, ax = plt.subplots(figsize=[10, 8])



bottom_teams_players = all_data[all_data['Club'].isin(bottom_teams)].copy()

bottom_teams_players['Club'] = bottom_teams_players['Club'].astype("category")

bottom_teams_players['Club'].cat.set_categories(bottom_teams[::-1], inplace=True)



sns.boxplot(data=bottom_teams_players, x='Overall', y='Club', showmeans=True)

ax.set_xlim([45,100])
fix, ax = plt.subplots(figsize=[10, 12], sharey=True)

sns.violinplot(data=bottom_teams_players, x='Overall', y='Club',

               inner='quartile', bw=.2, ax=ax)

ax.set_xlim([45, 100])
fix, ax = plt.subplots(figsize=[12, 10])



stats = ['Overall', 'Potential', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',

         'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',

         'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',

         'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions',

         'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',

         'Positioning', 'Reactions', 'Short passing', 'Shot power',

         'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

         'Strength', 'Vision', 'Volleys']



def left_of_operator(s):

    if any(x in str(s) for x in ('+', '-')):

        return re.split(r'\+|\-', s)[0]

    else:

        return s



for s in stats:

    all_data[s] = all_data[s].apply(left_of_operator).astype(int)



correlation = all_data[stats].corr()



sns.heatmap(correlation)
preferred_positions = all_data['Preferred Positions'].unique().tolist()

#print (preferred_positions)

positions=set(x for p in preferred_positions for x in p.split(' ') if x)
fix, ax = plt.subplots(len(positions), figsize=[10, 8], sharex=True)



best_by_position = []



for i, pos in enumerate(positions):

    matches = all_data[all_data['Preferred Positions'].str.contains(pos, case=False)].copy()

    matches['Position'] = pos

    sns.boxplot(data=matches, x='Overall', y='Position', ax=ax[i], showmeans=True)
fix, ax = plt.subplots(figsize=[10, 8], sharex=True)



best_by_position = []



for pos in positions:

    matches = all_data[all_data['Preferred Positions'].str.contains(pos, case=False)].copy()

    best_at_pos = matches.nlargest(1, columns='Overall').iloc[0].to_dict()

    best_at_pos['Best At'] = pos

    best_by_position.append(best_at_pos)

best_by_position = pd.DataFrame.from_records(best_by_position)

sns.barplot(data=best_by_position, y='Best At', x='Overall', label='Name')
pos_cols = ['CAM', 'CB', 'CDM', 'CF', 'CM', 'LAM', 'LB', 'LCB', 

            'LCM', 'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB', 'RAM', 

            'RB', 'RCB', 'RCM', 'RDM', 'RF', 'RM', 'RS', 'RW', 

            'RWB', 'ST']



fix, ax = plt.subplots(figsize=[12, 10])

correlation = all_data[pos_cols].corr()



sns.heatmap(correlation)

plt.xticks(rotation=45)
def convert_value(v):

    lookups = {'K': 1000, 'M': 1000000}

    value = v.replace('€', '')

    if value[-1] in ('K', 'M'):

        return float(value[:-1]) * lookups[value[-1]]

    else:

        return float(value)



all_data['Float_Wage'] = all_data['Wage'].apply(convert_value)

all_data['Float_Value'] = all_data['Value'].apply(convert_value)
fig, ax = plt.subplots(2, figsize=(10, 8), sharex=True)



all_data.plot(kind='scatter', x='Overall', y='Float_Value', ax=ax[0])

all_data.plot(kind='scatter', x='Overall', y='Float_Wage', ax=ax[1])



ax[0].get_yaxis().get_major_formatter().set_scientific(False)
cheap_good = all_data[(all_data['Float_Value'] <= 5000000) &

                      (all_data['Overall'] >= 80)]

cheap_good[['Name', 'Age', 'Club', 'Overall', 'Value', 'Wage']]
all_data['Speediness'] = (all_data['Acceleration'] + all_data['Sprint speed'])/2

fastest = pd.concat((all_data.nlargest(15, columns='Speediness'),

                     all_data.nsmallest(15, columns='Speediness')))

fastest.sort_values(by='Speediness', ascending=True, inplace=True)

fastest.set_index('Name', inplace=True)



fix, ax = plt.subplots(figsize=[10, 8])



fastest[['Acceleration', 'Sprint speed']].plot(kind='barh', stacked=False, ax=ax)

plt.legend(frameon=True, facecolor='white', 

           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))

ax.set_xlim([0, 100])

all_data['Weird_speed'] = abs(all_data['Acceleration'] - all_data['Sprint speed'])

weirdest = all_data.nlargest(15, columns='Weird_speed')

weirdest.sort_values(by='Weird_speed', ascending=True, inplace=True)

weirdest.set_index('Name', inplace=True)



fix, ax = plt.subplots(figsize=[10, 8])



weirdest[['Acceleration', 'Sprint speed']].plot(kind='barh', stacked=False, ax=ax)

plt.legend(frameon=True, facecolor='white', 

           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))

ax.set_xlim([0, 100])