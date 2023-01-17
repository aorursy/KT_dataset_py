import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../input/all_seasons.csv')
df.tail(10)
# Reset game number for season 9.
# It would be faster to use range() to create the new Game numbers, but there might be instances where games were skipped/not recorded
df.update(df.loc[df['season']==9, 'Game #'] - df.loc[df['season']==9, 'Game #'].min() + 1)

colours = ['r', 'k', 'b']

fig, ax = plt.subplots(nrows=1, ncols=1)
for s, c in zip(df['season'].unique(), colours):
    x = df[df['season'] == s]['Game #']
    y = df[df['season'] == s]['End SR']
    
    ax.plot(x, y, color=c, alpha=0.3, linewidth=1, label='')
    
    fit = np.polyfit(x, y, deg=1)
    ax.plot(x, fit[0] * x + fit[1], color=c, label='Season {}'.format(s), linestyle='--')

ax.set_xlabel('Game ID')
ax.set_ylabel('Skill Rating')
plt.title('Skill rating over time')
plt.legend()
plt.show()
# get roles counts, even if there were multiple per game
role1_counts = df[df['season'] == 9]['Role 1'].value_counts()
role2_counts = df[df['season'] == 9]['Role 2'].value_counts()

role_counts = role1_counts + role2_counts

plt.bar(role_counts.index, role_counts)
plt.xlabel('Role')
plt.ylabel('# of games played')
plt.title('Number of games for each role')
plt.show()
r1 = df[df['Role 1'].notnull()][['Role 1', 'Result']].rename(columns={'Role 1': 'Role'})
r2 = df[df['Role 2'].notnull()][['Role 2', 'Result']].rename(columns={'Role 2': 'Role'})
win_loss = r1.append(r2)

xtab = pd.crosstab(win_loss['Role'], win_loss['Result']).apply(lambda r: r/r.sum(), axis=1)

plt.bar(xtab.index, xtab['Win']*100)
plt.xlabel('Role')
plt.ylabel('Win %')
plt.title('Win percentage for each role')
plt.show()
# convert match time to minutes
df['match_time_split'] = df['Match Time'].str.split(':')
mins = df[df['match_time_split'].notnull()]['match_time_split'].str[0].astype(float)
secs = (df[df['match_time_split'].notnull()]['match_time_split'].str[1].astype(float)/60)
df['match_mins'] = mins + secs

mode_time = df[['Map', 'Mode', 'match_mins']].groupby('Mode').mean()

plt.bar(mode_time.index, mode_time['match_mins'])
plt.title('Average match length for each game mode')
plt.xlabel('Game Mode')
plt.ylabel('Match Duration (mins)')
plt.show()
df['Map'].value_counts().plot(kind='bar', color='b', title='# of games on each map')
map_time = df[['Map', 'Mode', 'match_mins']].groupby(['Map','Mode']).mean().reset_index()
map_time = map_time.sort_values('match_mins').reset_index(drop=True)
map_time['color'] = map_time['Mode'].replace({'Control': 'r',
                                              'Escort': 'g',
                                              'Assault': 'b',
                                              'Assault/Escort': 'k'})

ind = range(len(map_time['Map'].unique()))

plt.bar(ind, map_time['match_mins'], color=map_time['color'])

c = mpatches.Patch(color='r', label='Control')
e = mpatches.Patch(color='g', label='Escort')
a = mpatches.Patch(color='b', label='Assault')
ae = mpatches.Patch(color='k', label='Assault/Escort')

plt.legend(handles=[c,e,a,ae], loc=2)
plt.xticks(ind, map_time['Map'], rotation='vertical')
plt.ylim(8,)
plt.ylabel('Match Duration (mins)')
plt.title('Average match duration for each map')

plt.show()
xtab = pd.crosstab(df['Mode'], df['Result']).apply(lambda r: r/r.sum(), axis=1)

plt.bar(xtab.index, xtab['Win']*100)
plt.xlabel('Game Mode')
plt.ylabel('Win %')
plt.title('Win percentage for each game mode')
plt.show()
df['Team SR avg'] = df['Team SR avg'].replace({'P': np.nan}).astype(float)
df['Enemy SR avg'] = df['Enemy SR avg'].replace({'P': np.nan}).astype(float)
df['SR diff'] = df['Team SR avg'] - df['Enemy SR avg']

cur_df = df[df['season']==9]
df_win = cur_df[cur_df['Result']=='Win']
df_loss = cur_df[cur_df['Result']=='Loss']

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,7))

ax[0].scatter(df_win['SR diff'], df_win['SR Change'], alpha=0.5, color='g')

idx = np.isfinite(df_win['SR diff']) & np.isfinite(df_win['SR Change'])
fit = np.polyfit(df_win['SR diff'][idx], df_win['SR Change'][idx], 1)

ax[0].plot(df_win['SR diff'], fit[0] * df_win['SR diff'] + fit[1], color='k')
ax[0].text(35, 25, r'$\beta$ = {:.3f}'.format(fit[0]))

ax[0].set_title('Wins')
ax[0].set_ylabel('SR gain')
ax[0].set_xlabel('SR difference\nPositive = my team has higher SR\nNegative = enemy team has higher SR')

ax[1].scatter(df_loss['SR diff'], df_loss['SR Change'].abs(), alpha=0.5, color='r')

idx = np.isfinite(df_loss['SR diff']) & np.isfinite(df_loss['SR Change'])
fit = np.polyfit(df_loss['SR diff'][idx], df_loss['SR Change'][idx].abs(), 1)

ax[1].plot(df_loss['SR diff'], fit[0] * df_loss['SR diff'] + fit[1], color='k')
ax[1].text(50, 24, r'$\beta$ = {:.3f}'.format(fit[0]))

ax[1].set_title('Losses')
ax[1].set_ylabel('SR loss')
ax[1].set_xlabel('SR difference\nPositive = my team has higher SR\nNegative = enemy team has higher SR')

plt.tight_layout()
plt.show()
cur_df = df[(df['season'] == 9) & (df['Result']=='Win')]

# get only the games where I played support
support_idx = (cur_df['Role 1'] == 'Support') | (cur_df['Role 2'] == 'Support')
cur_df = cur_df[support_idx]
heal_medal = cur_df.groupby('Heal_medal')['SR Change'].mean().sort_values()

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6,9))

ax[0].scatter(cur_df['Heal'], cur_df['SR Change'], alpha=0.5)
fit = np.polyfit(cur_df['Heal'], cur_df['SR Change'], deg=1)
ax[0].plot(cur_df['Heal'], fit[0] * cur_df['Heal'] + fit[1], color='k')
ax[0].set_xlabel('Amount of healing done')
ax[0].set_ylabel('SR gain')

ax[1].scatter(cur_df['Heal'] - cur_df['Heal_career'], cur_df['SR Change'], alpha=0.5)
fit = np.polyfit(cur_df['Heal'] - cur_df['Heal_career'], cur_df['SR Change'], deg=1)
ax[1].plot(cur_df['Heal'] - cur_df['Heal_career'], fit[0] * (cur_df['Heal'] - cur_df['Heal_career']) + fit[1], color='k')
ax[1].set_xlabel('Healing done relative to total career healing')
ax[1].set_ylabel('SR gain')

ax[2].bar([0,1,2,3], heal_medal)
ax[2].set_yticks(range(0,35,5))
ax[2].set_xticks([0,1,2,3])
ax[2].set_xticklabels(heal_medal.index)
ax[2].set_xlabel('Healing medal received')
ax[2].set_ylabel('Average SR gain')

plt.tight_layout()
plt.show()
cur_df = df[(df['season'] == 9) & (df['Result']=='Win')]
idx = np.isfinite(cur_df['Death']) & np.isfinite(cur_df['SR Change'])
cur_df = cur_df[idx]

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.scatter(cur_df['Death'], cur_df['SR Change'])
fit = np.polyfit(cur_df['Death'], cur_df['SR Change'], deg=1)
ax.plot(cur_df['Death'], fit[0] * cur_df['Death'] + fit[1], color='k')
ax.set_xlabel('Number of deaths')
ax.set_ylabel('SR gain')

plt.tight_layout()
plt.show()
cur_df = df[(df['season'] == 9) & (df['Result']=='Win')]
idx = np.isfinite(cur_df['Death']) & np.isfinite(cur_df['SR Change'])
cur_df = cur_df[idx]

fig, ax = plt.subplots(nrows=1, ncols=1)

p = ax.scatter(cur_df['Death'], cur_df['SR Change'], c=cur_df['Heal'], cmap='gray')
ax.set_xlabel('Number of deaths')
ax.set_ylabel('SR gain')

plt.tight_layout()
cb = plt.colorbar(p)
cb.set_label('Healing done')
plt.show()
cur_df = df[(df['season'] == 9) & (df['Result']=='Win')]
idx = np.isfinite(cur_df['Death']) & np.isfinite(cur_df['SR Change'])
cur_df = cur_df[idx]

fig, ax = plt.subplots(nrows=1, ncols=1)

p = ax.scatter(cur_df['Death'], cur_df['SR Change'], c=cur_df['match_mins'], cmap='gray')
ax.set_xlabel('Number of deaths')
ax.set_ylabel('SR gain')

plt.tight_layout()
cb = plt.colorbar(p)
cb.set_label('Match duration (mins)')
plt.show()
# calculate deaths per minute
df['deaths_per_min'] = df['Death'] / df['match_mins']

cur_df = df[(df['season'] == 9) & (df['Result']=='Win')]
idx = np.isfinite(cur_df['deaths_per_min']) & np.isfinite(cur_df['SR Change'])
cur_df = cur_df[idx]

fig, ax = plt.subplots(nrows=1, ncols=1)

p = ax.scatter(cur_df['deaths_per_min'], cur_df['SR Change'], c=cur_df['Heal'], cmap='gray')
fit = np.polyfit(cur_df['deaths_per_min'], cur_df['SR Change'], deg=1)
ax.plot(cur_df['deaths_per_min'], fit[0] * cur_df['deaths_per_min'] + fit[1], color='r')
ax.set_xlabel('Deaths per minute')
ax.set_ylabel('SR gain')

plt.tight_layout()
cb = plt.colorbar(p)
cb.set_label('Healing done')
plt.show()