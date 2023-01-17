import pandas as pd



import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
# Load the original data (index and drop as there's a dummy index I don't want to keep).

master_demos = pd.read_csv('../input/csgofixed/mm_master_demos.csv', index_col=0).reset_index(drop=True)
# Group by `file` and `round` and take the first entry to avoid duplicates.  This gives us a row for each pair.

master_by_round = master_demos.groupby(['file', 'round'], as_index=False).first()



# Group again by `file`, which lets us then extract repeated information for all rounds for a given file.

master_by_file = master_by_round.groupby('file')



# Extract the first occurrence of `map` per file.

map_by_file = master_by_file['map'].first().reset_index()



# Extract and tally `winner_side` per file.  CT/T values may not always be present, so fill those with 0.

side_per_file = master_by_file['winner_side'].value_counts().reset_index(name='amount')

side_per_file = side_per_file.pivot_table(index='file', columns='winner_side', values='amount', fill_value=0).reset_index()

side_per_file.columns = ['file', 'ct_wins', 't_wins']



# Merge the data together per file.

per_game_df = map_by_file.merge(side_per_file, on='file')



# Cleanup.

del master_by_round

del master_by_file

del map_by_file

del side_per_file



per_game_df.head()
df = pd.DataFrame(per_game_df[['ct_wins', 't_wins']].sum().values, columns=['amount'], index=['Counter Terrorists', 'Terrorists'])

sns.barplot(data=df, x=df.index, y='amount')

sns.despine()

plt.xlabel('Winning Side')

plt.ylabel('Total Wins')

plt.title('Total wins per side')

plt.show()



del df
wins_per_map = per_game_df.rename({'ct_wins': 'Counter Terrorist', 't_wins': 'Terrorist'}, axis=1).groupby('map').sum().reset_index()

wins_per_map = wins_per_map.melt(id_vars='map').rename({'variable': 'side', 'value': 'amount'}, axis=1)

_ = plt.figure(figsize=(10, 10))

g = sns.barplot(data=wins_per_map, x='map', y='amount', hue='side')

sns.despine()

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.xlabel('Map')

plt.ylabel('Wins')

plt.title('Wins per side per map')

plt.show()



del wins_per_map

del g
def wins_per_map(df, map_name, ax):

    # Filter out the map we want.

    map_df = df.loc[df.map == map_name]

    # Sum again but based on this new df.

    wins_df = pd.DataFrame(map_df[['ct_wins', 't_wins']].sum().values, columns=['amount'], index=['Counter Terrorists', 'Terrorists'])

    # Plot!

    sns.barplot(data=wins_df, x=wins_df.index, y='amount', ax=ax)

    sns.despine()

    ax.set(title=map_name, ylabel='Total Wins')

    return wins_df



fig, axes = plt.subplots(7, 3, figsize=(15, 15), sharex=False)

fig.subplots_adjust(hspace=0.5)

for ax, curr_map in zip(axes.flatten(), sorted(per_game_df.map.unique())):

    wins_per_map(per_game_df, curr_map, ax)

fig.tight_layout()
top_5_weaps = master_demos.wp.value_counts().head(5)

sns.barplot(x=top_5_weaps.index, y=top_5_weaps.values)

sns.despine()

plt.xlabel('Weapon')

plt.ylabel('Total occurrences')

plt.title('Five most popular weapons overall')

plt.show()



del top_5_weaps
worst_5_weaps = master_demos.wp.value_counts().tail(5)

sns.barplot(x=worst_5_weaps.index, y=worst_5_weaps.values)

sns.despine()

plt.xlabel('Weapon')

plt.ylabel('Total occurrences')

plt.title('Five least popular weapons overall')

plt.show()



del worst_5_weaps
# Group by `map` and `wp` then see how many there are.  This gives us an entry for each map/wp combination and the total number of occurrences.

weap_by_map = master_demos.groupby(['map', 'wp'], as_index=False).size().reset_index().rename({0: 'amount'}, axis=1)



for curr_map in weap_by_map['map'].unique():

    print(f'{curr_map} top 5 weapons:')

    curr_map_weap = weap_by_map.loc[weap_by_map['map'] == curr_map]

    for curr_weap in curr_map_weap.sort_values('amount', ascending=False)[:5].iterrows():

        print(f'\t{curr_weap[1][1]} ({curr_weap[1][2]})')

    print()

    

del weap_by_map
g = sns.countplot(master_demos['hitbox'], order=master_demos['hitbox'].value_counts().index)

sns.despine()

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.xlabel('Hitbox location')

plt.ylabel('Amount')

plt.title('Most common hitboxes')

plt.show()



del g
sniper_box = master_demos.loc[master_demos['wp_type'] == 'Sniper', 'hitbox']

g = sns.countplot(sniper_box, order=sniper_box.value_counts().index)

sns.despine()

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.xlabel('Hitbox location')

plt.ylabel('Amount')

plt.title('Most common hitboxes - Sniper')

plt.show()



del sniper_box

del g
pistol_box = master_demos.loc[master_demos['wp_type'] == 'Pistol', 'hitbox']

g = sns.countplot(pistol_box, order=pistol_box.value_counts().index)

sns.despine()

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.xlabel('Hitbox location')

plt.ylabel('Amount')

plt.title('Most common hitboxes - Pistol')

plt.show()



del pistol_box

del g
import math



def convert_x_to_map(start_x, size_x, res_x, x):

    x += (start_x * -1.0) if (start_x < 0) else start_x

    x = math.floor((x / size_x) * res_x)

    return x



def convert_y_to_map(start_y, size_y, res_y, y):

    y += (start_y * -1.0) if (start_y < 0) else start_y

    y = math.floor((y / size_y) * res_y)

    y = (y - res_y) * -1.0

    return y



map_data = pd.read_csv('../input/csgofixed/map_data.csv', index_col=0)

map_data.loc['de_overpass'] = {'StartX': -4820, 'StartY': -3591, 'EndX': 503, 'EndY': 1740, 'ResX': 1024, 'ResY': 1024}

map_data.loc['de_nuke'] = {'StartX': -3082, 'StartY': -4464, 'EndX': 3516, 'EndY': 2180, 'ResX': 1024, 'ResY': 1024}

map_data
# Create the columns that will contain the converted coordinates.

master_demos['AttackPosX'] = np.nan

master_demos['AttackPosY'] = np.nan

master_demos['VictimPosX'] = np.nan # I think this is for victim position?

master_demos['VictimPosY'] = np.nan





for map_name in master_demos['map'].unique():

    if map_name not in map_data.index:

        print(f'Data not found for map: {map_name}')

        continue

    # Pull metadata for the map in question.

    data = map_data.loc[map_name]

    start_x = data['StartX']

    start_y = data['StartY']

    end_x = data['EndX']

    end_y = data['EndY']

    size_x = end_x - start_x

    size_y = end_y - start_y

    res_x = data['ResX']

    res_y = data['ResY']

    

    # Apply the conversion functions to the appropriate columns and store them in the dummy columns created earlier.

    print(f'Converting coordinates for {map_name}', end='')

    master_demos.loc[master_demos['map'] == map_name, 'AttackPosX'] =  master_demos.loc[master_demos['map'] == map_name, 'att_pos_x'].apply(lambda x: convert_x_to_map(start_x, size_x, res_x, x))

    master_demos.loc[master_demos['map'] == map_name, 'AttackPosY'] =  master_demos.loc[master_demos['map'] == map_name, 'att_pos_y'].apply(lambda y: convert_y_to_map(start_y, size_y, res_y, y))

    master_demos.loc[master_demos['map'] == map_name, 'VictimPosX'] =  master_demos.loc[master_demos['map'] == map_name, 'vic_pos_x'].apply(lambda x: convert_x_to_map(start_x, size_x, res_x, x))

    master_demos.loc[master_demos['map'] == map_name, 'VictimPosY'] =  master_demos.loc[master_demos['map'] == map_name, 'vic_pos_y'].apply(lambda y: convert_y_to_map(start_y, size_y, res_y, y))

    print('...done!')



# Cleanup.

del map_data
map_name = 'de_dust2'



dust_data = master_demos.loc[master_demos['map'] == map_name]



# Plot attack positions.

plt.figure(figsize=(20, 20))

plt.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))

plt.scatter(dust_data['AttackPosX'], dust_data['AttackPosY'], alpha=0.005, c='blue')

plt.title(f'Attacker positions for {map_name}', fontsize=20)

plt.show()



# Plot victim positions.

plt.figure(figsize=(20, 20))

plt.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))

plt.scatter(dust_data['VictimPosX'], dust_data['VictimPosY'], alpha=0.005, c='red')

plt.title(f'Victim positions for {map_name}', fontsize=20)

plt.show()
dust_sniper_data = dust_data.loc[dust_data['wp_type'] == 'Sniper']



plt.figure(figsize=(20, 20))

plt.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))

plt.scatter(dust_sniper_data['AttackPosX'], dust_sniper_data['AttackPosY'], alpha=0.01, c='blue')

plt.scatter(dust_sniper_data['VictimPosX'], dust_sniper_data['VictimPosY'], alpha=0.01, c='red')

plt.title(f'Attacker and Victim positions for {map_name} with snipers', fontsize=20)

plt.show()
filtered_dust_sniper_data = dust_sniper_data.loc[

    (

        (dust_sniper_data.AttackPosX > 400) &

        (dust_sniper_data.AttackPosX < 500) &

        (dust_sniper_data.AttackPosY > 900)

    )

]



plt.figure(figsize=(20, 20))

plt.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))

plt.scatter(filtered_dust_sniper_data['AttackPosX'], filtered_dust_sniper_data['AttackPosY'], alpha=0.01, c='blue')

plt.scatter(filtered_dust_sniper_data['VictimPosX'], filtered_dust_sniper_data['VictimPosY'], alpha=0.01, c='red')

plt.title(f'{map_name} Terrorist spawn sniper shots', fontsize=20)

plt.show()
from matplotlib.collections import LineCollection



rays_origin = list(zip(filtered_dust_sniper_data.AttackPosX, filtered_dust_sniper_data.AttackPosY))

rays_dest = list(zip(filtered_dust_sniper_data.VictimPosX, filtered_dust_sniper_data.VictimPosY))

lines_formatted = list(map(lambda x, y: [x, y], rays_origin, rays_dest))

lc = LineCollection(lines_formatted, linestyles='dashed', colors=[(1, 0, 0, 0.05)])



fig, ax = plt.subplots(1, figsize=(20, 20))

ax.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))

ax.add_collection(lc)

plt.title(f'{map_name} Terrorist spawn sniper shots', fontsize=20)

plt.show()



del rays_origin

del rays_dest

del lines_formatted

del lc

del fig

del ax
# A bit of cleanup before continuing.

del filtered_dust_sniper_data

del dust_sniper_data
# If true, will run the animation process. READ THE WARNING FIRST.

do_animation = False



# If true, will save the animation out.

save_animation = False
from IPython.display import HTML

from matplotlib import animation, rc



if do_animation:

    # Filter out a particular round.

    round_data = dust_data.loc[dust_data['round'] == 1]



    # How many encounters to show at the same time?

    window_size = 100

    # How many sliding windows will we have?

    window_count = len(round_data) - window_size



    # Create and configure the figure.

    fig, ax = plt.subplots(1, figsize=(10, 10))

    title = ax.text(0.5, 0.95, 'ayyy lmao', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, transform=ax.transAxes, ha='center')

    image = ax.imshow(plt.imread(f'../input/csgofixed/{map_name}.png'))

    attack_scatter = ax.scatter([], [], alpha=1.0, c='blue')

    victim_scatter = ax.scatter([], [], alpha=1.0, c='red', marker='x')



    # Animation.

    def update(i):

        attack_subset = list(zip(round_data['AttackPosX'][i:i+window_size], round_data['AttackPosY'][i:i+window_size]))

        attack_scatter.set_offsets(attack_subset)



        victim_subset = list(zip(round_data['VictimPosX'][i:i+window_size], round_data['VictimPosY'][i:i+window_size]))

        victim_scatter.set_offsets(victim_subset)



        title.set_text(f'{map_name} Attacks & Victims {i} -> {i+window_size}')



        return (title,)



    anim = animation.FuncAnimation(fig, update, interval=20, frames=window_count, blit=True)
# Export out the video into a HTML video.

if do_animation:

    HTML(anim.to_html5_video())
if do_animation and save_animation:

    writer_ffmpeg = animation.writers['ffmpeg']

    writer = writer_ffmpeg(fps=15, bitrate=1800)

    anim.save('anim.mp4', writer=writer)