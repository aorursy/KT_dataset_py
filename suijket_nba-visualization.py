# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage import io

import matplotlib.pyplot as plt

# import libraries

%matplotlib inline







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# loading dataset

nba = pd.read_csv('/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv')

# show first five rows

nba.head()
nba.describe()
# Set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# Set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# Create dataframe

nba_RegularSeason = nba.loc[nba['Stage'] == 'Regular_Season']

nba_3 = nba_RegularSeason[['Season', '3PA']].groupby('Season').sum()



# Numeric placeholder for the y axis

my_range = list(range(1,len(nba_3.index)+1))

fig, ax = plt.subplots(figsize=(8,4))





# create for each expense type an horizontal line that starts at x = 0 with the length 

# represented by the specific expense percentage value.

plt.hlines(y=my_range, xmin=0, xmax=nba_3['3PA'], color='#0d67a3', alpha=0.4, linewidth=7)



# create for each expense type a dot at the level of the expense percentage value

plt.plot(nba_3['3PA'], my_range, "o", markersize=8, color='#0d67a3', alpha=0.6)



# set labels

ax.set_xlabel("Number of 3P's attempted", fontsize=13, fontweight='black', color = '#333F4B')

ax.set_ylabel("Season", fontsize=13, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.yticks(my_range, nba_3.index)



# add an horizonal label for the y axis 

fig.text(-0.23, 0.96, "Number of 3P's made per Season", fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



#plt.savefig('hist2.png', dpi=300, bbox_inches='tight')
def get_percentage(list_a, list_m):

    return list_a / list_m



nba['2PM'] = nba['FGM'] - nba['3PM']

nba['2PA'] = nba['FGA'] - nba['3PA']

nba['Avg_Min'] = nba['MIN'] / nba['GP']

nba['Avg_AST'] = nba['AST'] / nba['GP']

nba['Avg_STL'] = nba['STL'] / nba['GP']

nba['Avg_BLK'] = nba['BLK'] / nba['GP']

nba['Avg_PTS'] = nba['PTS'] / nba['GP']

nba['Avg_3P'] = nba['3PM'] / nba['GP']

nba['Avg_2P'] = nba['2PM'] / nba['GP']

nba['Avg_FT'] = nba['FTM'] / nba['GP']

nba['Avg_REB'] = nba['REB'] / nba['GP']



nba['percentage_2P'] = get_percentage(nba['2PM'], nba['2PA'])

nba['percentage_3P'] = get_percentage(nba['3PM'], nba['3PA']) 

nba['percentage_FT'] = get_percentage(nba['FTM'], nba['FTA'])

nba['percentage_OR'] = get_percentage(nba['ORB'], nba['REB'])

nba['percentage_DR'] = get_percentage(nba['DRB'], nba['REB'])
# Set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# Set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# Create dataframe

nba_RegularSeason = nba.loc[nba['Stage'] == 'Regular_Season']

nba_percentage = nba_RegularSeason[['Season', 'percentage_3P']].groupby('Season').mean()



# Numeric placeholder for the y axis

my_range = list(range(1,len(nba_percentage.index)+1))

fig, ax = plt.subplots(figsize=(8,4))



# create for each expense type an horizontal line that starts at x = 0 with the length 

# represented by the specific expense percentage value.

plt.hlines(y=my_range, xmin=0, xmax=nba_percentage['percentage_3P'], color='#0d67a3', alpha=0.4, linewidth=7)



# create for each expense type a dot at the level of the expense percentage value

plt.plot(nba_percentage['percentage_3P'], my_range, "o", markersize=8, color='#0d67a3', alpha=0.6)



# set labels

ax.set_xlabel("Percentage", fontsize=13, fontweight='black', color = '#333F4B')

ax.set_ylabel("Season", fontsize=13, fontweight='black', color = '#333F4B')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.yticks(my_range, nba_percentage.index)



# add an horizonal label for the y axis 

fig.text(-0.23, 0.96, "Percentage 3P's made per Season", fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))



#plt.savefig('hist2.png', dpi=300, bbox_inches='tight')
def plot_img(url, idx):

    # read and plot img

    image = io.imread(url)

    plt.imshow(image)

    

    # plot title n remove axis

    plt.title(nba_mvp.Player[idx], loc='center', fontsize=18)

    ax.axis('off')

    

def plot_radar(idx, stage, color_1):

    # categories

    categories = nba_mvp.iloc[idx,2:].index.tolist()

    N = len(categories) # get number of categories

    

    # values

    values= nba_mvp.iloc[idx,2:].values.tolist()

    values += values[:1] # repeat first value to close poly

    # calculate angle for each category

    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    angles += angles[:1] # repeat first angle to close poly

    # plot

    plt.polar(angles, values, marker='.', label=stage, color=color_1) # lines

    plt.fill(angles, values, alpha=0.2, color = color_1) # area

    

    # xticks

    plt.xticks(angles[:-1], categories)

    # yticks

    ax.set_rlabel_position(0) # yticks position

    plt.yticks([0, 2, 4, 6, 8, 10, 12], color="grey", size=10)

    plt.ylim(0,14)
def plot_img(url, idx):

    # read and plot img

    image = io.imread(url)

    plt.imshow(image)

    

    # plot title n remove axis

    plt.title(nba_mvp.Player[idx], loc='center', fontsize=18)

    ax.axis('off')

    

def plot_radar(idx, stage, color_1):

    # categories

    categories = nba_mvp.iloc[idx,2:].index.tolist()

    N = len(categories) # get number of categories

    

    # values

    values= nba_mvp.iloc[idx,2:].values.tolist()

    values += values[:1] # repeat first value to close poly

    # calculate angle for each category

    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    angles += angles[:1] # repeat first angle to close poly

    # plot

    plt.polar(angles, values, marker='.', label=stage, color=color_1) # lines

    plt.fill(angles, values, alpha=0.2, color = color_1) # area

    

    # xticks

    plt.xticks(angles[:-1], categories)

    # yticks

    ax.set_rlabel_position(0) # yticks position

    plt.yticks([0, 2, 4, 6, 8, 10, 12], color="grey", size=10)

    plt.ylim(0,14)
idx1_regular = 1272 # 

idx1_playoff = 3197 # 



idx2_regular = 1468 # 

idx2_playoff = 3363



idx3_regular = 1756 # 

idx3_playoff = 3500



idx4_regular = 2040

idx4_playoff = 3656



nba_mvp = nba[['Player', 'Avg_AST', 'Avg_STL', 'Avg_BLK', 'Avg_3P', 'Avg_2P', 'Avg_FT', 'Avg_REB']]

fig = plt.figure(figsize=(12,8))

plt.subplots_adjust(right = 1.5)



# img 1

ax = plt.subplot(241)

plot_img('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3975.png&w=350&h=254', idx1_regular)

# img 2

ax = plt.subplot(242)

plot_img('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3468.png&w=350&h=254', idx2_regular)

# img 3

ax = plt.subplot(243)

plot_img('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3992.png&w=350&h=254', idx3_regular)

# img 6

ax = plt.subplot(244)

plot_img('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3032977.png&w=350&h=254', idx4_regular)





### Second Row

# radar 1

ax = plt.subplot(245, polar="True")

plot_radar(idx1_regular, 'regular', 'yellow')

plot_radar(idx1_playoff, 'playoff', 'blue')

ax.legend()



# radar 2

ax = plt.subplot(246, polar="True")

plot_radar(idx2_regular, 'regular', 'red')

plot_radar(idx2_playoff, 'playoff', 'black')

ax.legend()



# radar 3

ax = plt.subplot(247, polar="True")

plot_radar(idx3_regular, 'regular', 'red')

plot_radar(idx3_playoff, 'playoff', 'black')

ax.legend()



# radar 3

ax = plt.subplot(248, polar="True")

plot_radar(idx4_regular, 'regular', 'green')

plot_radar(idx4_playoff, 'playoff', 'black')

ax.legend()



plt.show()
def plot_img(url, idx):

    # read and plot img

    image = io.imread(url)

    plt.imshow(image)

    

    # plot title n remove axis

    plt.title(nba.Team[idx], loc='center', fontsize=18)

    ax.axis('off')

    

def plot_radar(values, color_1):

    # categories

    categories = ['Avg_AST', 'Avg_STL', 'Avg_BLK', 'Avg_3P', 'Avg_2P', 'Avg_FT', 'Avg_REB']

    N = len(categories) # get number of categories

    

    # values

    #values = dataset.iloc[idx,2:].values.tolist()

    values += values[:1] # repeat first value to close poly

    # calculate angle for each category

    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    angles += angles[:1] # repeat first angle to close poly

    # plot

    plt.polar(angles, values, marker='.', color=color_1) # lines

    plt.fill(angles, values, alpha=0.2, color = color_1) # area

    

    # xticks

    plt.xticks(angles[:-1], categories)

    # yticks

    ax.set_rlabel_position(0) # yticks position

    plt.yticks([0, 2, 4, 6, 8, 10, 12], color="grey", size=10)

    plt.ylim(0,14)
years = ['2015 - 2016', '2016 - 2017', '2017 - 2018', '2018 - 2019']

nba_teams = ['CLE', 'GSW', 'GSW', 'TOR']

plot_number = [245, 246, 247, 248]

color = ['red', 'blue', 'blue', 'red']

avg_mins = []



count = 0

for year in years:

    test = nba.groupby(['Team', 'Season', 'Stage']).mean().reset_index()

    test_regular = int(test.loc[(test['Stage'] == 'Playoffs') &

                                (test['Team'] == nba_teams[count]) &

                                (test['Season'] == year)]['MIN'].values)

    avg_mins.append(test_regular)

    count += 1
fig = plt.figure(figsize=(12,8))

plt.subplots_adjust(right = 1.5)



# img 1

ax = plt.subplot(241)

plot_img('https://a.espncdn.com/i/teamlogos/nba/500/cle.png', 1)

# img 2

ax = plt.subplot(242)

plot_img('https://a.espncdn.com/i/teamlogos/nba/500/gs.png', 28)

# img 3

ax = plt.subplot(243)

plot_img('https://a.espncdn.com/i/teamlogos/nba/500/gs.png', 28)

# img 6

ax = plt.subplot(244)

plot_img('https://a.espncdn.com/i/teamlogos/nba/500/tor.png', 8)





for i in range(0,4):

    test_regular = nba.loc[(nba['Stage'] == 'Playoffs') &

                            (nba['Team'] == nba_teams[i]) &

                            (nba['Season'] == years[i]) &

                            (nba['MIN'] > avg_mins[i])]

    test_regular = test_regular[['Avg_AST', 'Avg_STL', 'Avg_BLK', 'Avg_3P', 'Avg_2P', 'Avg_FT', 'Avg_REB']]

    ax = plt.subplot(plot_number[i], polar="True")

    for j in test_regular.values.tolist():

        plot_radar(j, color[i])

        