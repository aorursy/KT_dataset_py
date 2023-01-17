import numpy as np 
import pandas as pd 
import os
import pickle
# I took the original data set and used riot's api to get more data about the games.
# The API has a request limit of 100 requests per 2 min, so I just have a pickle after I made a request for every game
df = pd.read_pickle('/kaggle/input/riotapi-pickles/riotapi_lower_res')

preproc_df = df.copy()

df.head()
df.columns
more_games_df = pd.read_pickle('/kaggle/input/riotapi-pickles/more_games_lower')

more_games_df.head(10)
# I'm dumb and basically had duplicate columns of these from the original data set and then from using the riot api
df.drop(['blue_firstBlood', 'red_firstBlood'], axis=1, inplace=True)

df['redWins'] = df['blueWins'].apply(lambda x: 1 if x == 0 else 0)
preproc_df['redWins'] = df['redWins']

more_games_df.dropna(inplace=True)
more_games_df.reset_index(inplace=True)
more_games_df.rename(columns={"redfirstBlood": 'redFirstBlood', 'bluefirstBlood': 'blueFirstBlood'}, inplace=True)
more_games_df['redWins'] = more_games_df['blueWins'].apply(
    lambda x: 1 if x == 0 else 0
)

# Copies the order of the original data set (I have obsessive compulsions regarding to order)
more_games_df = more_games_df[df.columns.tolist()]

# Combining the two temporarily for part of the preprocessing
combined_df = pd.concat([preproc_df, more_games_df], axis=0, ignore_index=True)
combined_df.drop(['blue_firstBlood', 'red_firstBlood'], axis=1, inplace=True)
combined_df.head()
blue_win = df[df['blueWins'] == 1]
red_win = df[df['blueWins'] == 0]
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot, plot
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)
fig = go.Figure()

blue_loss = df[df['blueWins'] == 0]

fig.add_trace(go.Bar(x=[0], y=list(blue_win['blueWins'].value_counts()), name='Blue', marker_color='#084177', width=0.5))
fig.add_trace(go.Bar(x=[1], y=list(blue_loss['blueWins'].value_counts()), name='Red',
                     marker_color=['#d63447'], width=0.5))

fig.update_layout(
    xaxis=dict(
        showticklabels=True,
        tickvals=[0, 1],
        ticktext=[i for i in ['Blue', 'Red']],
    ),
    yaxis_title='Wins',
    title='Wins From Each Team',
    height=800,
    width=800
)

iplot(fig)
fig = go.Figure(data=[
    go.Box(name='Blue Win', y=blue_win['blueWardsPlaced'], boxmean=True),
    go.Box(name='Blue Loss', y=red_win['blueWardsPlaced'], boxmean=True),
    go.Box(name='Red Win', y=red_win['redWardsPlaced'], boxmean=True),
    go.Box(name='Red Loss', y=blue_win['redWardsPlaced'], boxmean=True)
])

fig.update_layout(
    title='Wards Placed Distribution',
    height=800,
    width=800
)

iplot(fig)
fig = go.Figure(data=[
    go.Histogram(name='Blue Win', x=blue_win['blueWardsDestroyed']),
    go.Histogram(name='Blue Loss', x=red_win['blueWardsDestroyed']),
    go.Histogram(name='Red Win', x=red_win['redWardsDestroyed']),
    go.Histogram(name='Red Loss', x=blue_win['redWardsDestroyed'])
])

fig.update_layout(
    title='Wards Destroyed Distribution',
    height=800,
    width=800
)

iplot(fig)
fig = go.Figure(data=[
    go.Bar(name='Blue Win', x=[0], y=[np.sum(blue_win['blueFirstBlood'])], width=0.5),
    go.Bar(name='Blue Loss', x=[1], y=[np.sum(red_win['blueFirstBlood'])], width=0.5),
    go.Bar(name='Red Win', x=[2], y=[np.sum(red_win['redFirstBlood'])], width=0.5),
    go.Bar(name='Red Loss', x=[3], y=[np.sum(blue_win['redFirstBlood'])], width=0.5)
])

fig.update_layout(
    title='The Importance of First Kills',
    height=800,
    width=800,
    xaxis=dict(
        tickvals=[i for i in range(4)],
        ticktext=[i for i in ['Blue Win', 'Blue Loss', 'Red Win', 'Red Loss']],
        showticklabels=True
    ),
)

iplot(fig)
fig = go.Figure(data=[
    go.Histogram(name='Blue Win', x=blue_win['blueKills']),
    go.Histogram(name='Blue Loss', x=red_win['blueKills']),
    go.Histogram(name='Red Win', x=red_win['redKills']),
    go.Histogram(name='Red Loss', x=blue_win['redKills'])
])

fig.update_layout(
    title='Distribution of Team Kills when Winning and Losing',
    height=800,
    width=800,
)

iplot(fig)
fig = go.Figure(data=[
    go.Bar(name='Blue Win', x=[0], y=[np.mean(blue_win['blueKills'])], width=0.5),
    go.Bar(name='Blue Loss', x=[1], y=[np.mean(red_win['blueKills'])], width=0.5),
    go.Bar(name='Red Win', x=[2], y=[np.mean(red_win['redKills'])], width=0.5),
    go.Bar(name='Red Loss', x=[3], y=[np.mean(blue_win['redKills'])], width=0.5)
])

fig.update_layout(
    title='Average Kills of Teams when Winning and Losing',
    height=800,
    width=800,
    xaxis=dict(
        tickvals=[i for i in range(4)],
        ticktext=[i for i in ['Blue Win', 'Blue Loss', 'Red Win', 'Red Loss']],
        showticklabels=False,
        title='Team'
    ),
)

iplot(fig)
fig = go.Figure(data=[
    go.Bar(name='Blue Win', x=[0], y=[np.mean(blue_win['blueDeaths'])], width=0.5),
    go.Bar(name='Blue Loss', x=[1], y=[np.mean(red_win['blueDeaths'])], width=0.5),
    go.Bar(name='Red Win', x=[2], y=[np.mean(red_win['redDeaths'])], width=0.5),
    go.Bar(name='Red Loss', x=[3], y=[np.mean(blue_win['redDeaths'])], width=0.5)
])

fig.update_layout(
    title='Average Deaths of Teams when Winning and Losing',
    height=800,
    width=800,
    xaxis=dict(
        tickvals=[i for i in range(4)],
        ticktext=[i for i in ['Blue Win', 'Blue Loss', 'Red Win', 'Red Loss']],
        showticklabels=False,
        title='Team'
    ),
)

iplot(fig)
fig = go.Figure(data=[
    go.Bar(name='Blue Win', x=[0], y=[np.mean(blue_win['blueAssists'])], width=0.5),
    go.Bar(name='Blue Loss', x=[1], y=[np.mean(red_win['blueAssists'])], width=0.5),
    go.Bar(name='Red Win', x=[2], y=[np.mean(red_win['redAssists'])], width=0.5),
    go.Bar(name='Red Loss', x=[3], y=[np.mean(blue_win['redAssists'])], width=0.5)
])

fig.update_layout(
    title='Average Assists of Teams when Winning and Losing',
    height=800,
    width=800,
    xaxis=dict(
        tickvals=[i for i in range(4)],
        ticktext=[i for i in ['Blue Win', 'Blue Loss', 'Red Win', 'Red Loss']],
        showticklabels=True
    ),
)

iplot(fig)
fig = go.Figure(data=[
    go.Histogram(name='Blue Win', x=blue_win['blueTowersDestroyed']),
    go.Histogram(name='Blue Loss', x=red_win['blueTowersDestroyed']),
    go.Histogram(name='Red Win', x=red_win['redTowersDestroyed']),
    go.Histogram(name='Red Loss', x=blue_win['redTowersDestroyed'])
])

fig.update_layout(
    title='Distribution of Towers Destroyed when Winning and Losing',
    height=800,
    width=800,
)

iplot(fig)
fig = go.Figure(data=[
    go.Bar(name='Blue Win', y=[0], x=[np.sum(blue_win['blueEliteMonsters'])], width=0.5, orientation='h'),
    go.Bar(name='Blue Loss', y=[1], x=[np.sum(red_win['blueEliteMonsters'])], width=0.5, orientation='h'),
    go.Bar(name='Red Win', y=[2], x=[np.sum(red_win['redEliteMonsters'])], width=0.5, orientation='h'),
    go.Bar(name='Red Loss', y=[3], x=[np.sum(blue_win['redEliteMonsters'])], width=0.5, orientation='h')
])

fig.update_layout(
    title='Epic Monsters Killed',
    height=800,
    width=800,
    yaxis=dict(
        tickvals=[i for i in range(4)],
        ticktext=[i for i in ['Blue Win', 'Blue Loss', 'Red Win', 'Red Loss']],
        showticklabels=True
    ),
)

iplot(fig)
fig = go.Figure(data=[
    go.Box(name='Blue Win', x=blue_win['blueTotalExperience']),
    go.Box(name='Blue Loss', x=red_win['blueTotalExperience']),
    go.Box(name='Red Win', x=red_win['redTotalExperience']),
    go.Box(name='Red Loss', x=blue_win['redTotalExperience'])
])

fig.update_layout(
    title='Total Experience Distrubtion of Champions',
    height=800,
    width=800,
)

iplot(fig)
fig = go.Figure(data=[
    go.Violin(name='Blue Win', y=blue_win['blueTotalGold'], meanline_visible=True),
    go.Violin(name='Blue Loss', y=red_win['blueTotalGold'], meanline_visible=True),
    go.Violin(name='Red Win', y=red_win['redTotalGold'], meanline_visible=True),
    go.Violin(name='Red Loss', y=blue_win['redTotalGold'], meanline_visible=True),
])

fig.update_layout(
    title='Gold on Winning and Losing Teams',
    height=800,
    width=800,
)

iplot(fig)
fig = go.Figure(data=[
    go.Box(name='Blue Win', x=blue_win['blueCSPerMin']),
    go.Box(name='Blue Loss', x=red_win['blueCSPerMin']),
    go.Box(name='Red Win', x=red_win['redCSPerMin']),
    go.Box(name='Red Loss', x=blue_win['redCSPerMin'])
])

fig.update_layout(
    title='CS Per Min Distribution',
    height=800,
    width=800,
)

iplot(fig)
def get_champions(ids):
    if not os.path.isfile('/kaggle/input/riotapi-pickles/champions_lower'):
        r = requests.get('http://ddragon.leagueoflegends.com/cdn/10.10.3216176/data/en_US/champion.json')

        response = r.json()

        champions_reformatted = {}

        for champion in response['data']:
            id = response['data'][champion]['key']

            champions_reformatted[int(id)] = 'Wukong' if champion == 'MonkeyKing' else champion

        with open('/kaggle/input/riotapi-pickles/champions_lower', 'wb') as file:
            pickle.dump(champions_reformatted, file)
    else:
        with open('/kaggle/input/riotapi-pickles/champions_lower', 'rb') as file:
            champions_reformatted = pickle.load(file)

    champions = []

    for id in ids:

        champions.append('None' if id == -1 else champions_reformatted[id])

    return champions[0] if len(champions) == 1 else champions

def format_champs(df, cols, head):
    champ_dict = {}

    for col in cols:
        champs = df[col].value_counts()

        champ_names = get_champions(champs.keys())
        freq = [i for i in champs]
        counter = 0
        for name in champ_names:

            champ_dict[name] = champ_dict.setdefault(name, 0) + freq[counter]

            counter += 1

    champ_dict = {k: v for k, v in sorted(champ_dict.items(), key=lambda item: item[1], reverse=True)}

    top_n = dict(list(champ_dict.items())[0:head])

    other = dict(list(champ_dict.items())[head+1:])

    other_total = 0
    top_n_total = 0

    for key, val in top_n.items():
        top_n_total += val

    for key, val in other.items():
        other_total += val

    other_total = np.abs(top_n_total-other_total)

    top_n['Other'] = other_total

    return top_n
bans = ['ban_1', 'ban_2', 'ban_3', 'ban_4', 'ban_5',
        'ban_6', 'ban_7', 'ban_8', 'ban_9', 'ban_10']
figs = []
for ban in bans:
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                        subplot_titles=('Blue {} {}'.format(ban[0:3], ban[4:]),
                                        'Red {} {}'.format(ban[0:3], ban[4:]))
    )

    row = 1

    blue_win_all_ban = blue_win[ban].value_counts()
    red_win_ban_all_ban = red_win[ban].value_counts()

    blue_win_top_10 = blue_win[ban].value_counts().head(20)
    red_win_top_10 = red_win[ban].value_counts().head(20)

    blue_win_other = np.abs(np.sum(blue_win_top_10) - np.sum(blue_win_all_ban))
    red_win_other = np.abs(np.sum(red_win_top_10) - np.sum(red_win_ban_all_ban))

    blue_vals = blue_win_top_10.values
    red_vals = red_win_top_10.values

    blue_vals = np.append(blue_vals, blue_win_other)
    red_vals = np.append(red_vals, red_win_other)

    blue_bans = get_champions(list(blue_win_top_10.keys()))
    red_bans = get_champions(list(red_win_top_10.keys()))

    fig.add_trace(go.Pie(
        name=ban,
        labels=blue_bans + ['Other'],
        values=blue_vals),
        row=1,
        col=1
    )

    fig.add_trace(go.Pie(
        name=ban,
        labels=red_bans + ['Other'],
        values=red_vals),
        row=1,
        col=2
    )

    fig.update_layout(
        height=600,
        width=800
    )
    
    figs.append(fig)
iplot(figs[0])
iplot(figs[1])
iplot(figs[2])
iplot(figs[3])
iplot(figs[4])
iplot(figs[5])
iplot(figs[6])
iplot(figs[7])
iplot(figs[8])
iplot(figs[9])
champs = ['blue_champ_1', 'blue_champ_2', 'blue_champ_3', 'blue_champ_4', 'blue_champ_5',
          'red_champ_1', 'red_champ_2', 'red_champ_3', 'red_champ_4', 'red_champ_5']

champs_formatted = format_champs(df, champs, 35)

fig = go.Figure(data=[
    go.Pie(
        labels=list(champs_formatted.keys()),
        values=list(champs_formatted.values())
    )
])

fig.update_layout(
    height=900,
    width=800,
    title='Most Frequently Selected Champions'
)

iplot(fig)
blue_champs = champs[0:5]
red_champs = champs[5:]

blue_win_champs_formatted = format_champs(blue_win, blue_champs, 25)
blue_lose_champs_formatted = format_champs(red_win, blue_champs, 25)

red_win_champs_formatted = format_champs(red_win, red_champs, 25)
red_lose_champs_formatted = format_champs(blue_win, red_champs, 25)

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                    subplot_titles=('Top 25 Blue Champion Selections (Win)',
                                    'Top 25 Red Champion Selections (Lose)')
                    )

fig.add_trace(
    go.Pie(
        name='Blue',
        labels=list(blue_win_champs_formatted.keys()),
        values=list(blue_win_champs_formatted.values())
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Pie(
        name='Red',
        labels=list(red_lose_champs_formatted.keys()),
        values=list(red_lose_champs_formatted.values())
    ),
    row=1,
    col=2
)

fig.update_layout(
    height=800,
    width=800
)

iplot(fig)
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                    subplot_titles=('Top 25 Red Champion Selections (Win)',
                                    'Top 25 Blue Champion Selections (Lose)')
                    )

fig.add_trace(
    go.Pie(
        name='Red',
        labels=list(red_win_champs_formatted.keys()),
        values=list(red_win_champs_formatted.values())
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Pie(
        name='Blue',
        labels=list(blue_lose_champs_formatted.keys()),
        values=list(blue_lose_champs_formatted.values())
    ),
    row=1,
    col=2
)

fig.update_layout(
    height=800,
    width=800
)
blue_win_finhibit = blue_win[blue_win['blue_firstInhibitor'] == 1]
blue_win_ninhibit = blue_win[blue_win['blue_firstInhibitor'] == 0]

red_win_finhibit = red_win[red_win['red_firstInhibitor'] == 1]
red_win_ninhibit = red_win[red_win['red_firstInhibitor'] == 0]

fig = go.Figure(data=[
    go.Pie(
        labels=['Blue Win with First Inhibitor', 'Blue Win without First Inhibitor'],
        values=[np.sum(blue_win_finhibit['blueWins']), np.sum(blue_win_ninhibit['blueWins'])]
    )
])

fig.update_layout(
    title='Blue Wins With and Without First Inhibitor',
    height=800,
    width=800
)

iplot(fig)
fig = go.Figure(data=[
    go.Pie(
        labels=['Red Win with First Inhibitor', 'Red Win without First Inhibitor'],
        values=[np.sum(red_win_finhibit['redWins']), np.sum(red_win_ninhibit['redWins'])]
    )
])

fig.update_layout(
    title='Red Wins With and Without First Inhibitor',
    height=800,
    width=800
)

iplot(fig)
fig = go.Figure(data=[
    go.Bar(name='Blue', x=[0], y=[np.sum(df['blue_firstBaron'])], width=0.5, marker_color='#084177'),
    go.Bar(name='Red', x=[1], y=[np.sum(df['red_firstBaron'])], width=0.5, marker_color='#d63447')
])

fig.update_layout(
    title='First Baron Count',
    xaxis=dict(
        tickvals=[i for i in range(2)],
        ticktext=['Blue', 'Red'],
        showticklabels=True
    ),
    width=800,
    height=800
)
blue_win_fbaron = blue_win[blue_win['blue_firstBaron'] == 1]
blue_win_nbaron = blue_win[blue_win['blue_firstBaron'] == 0]

red_win_fbaron = red_win[red_win['red_firstBaron'] == 1]
red_win_nbaron = red_win[red_win['red_firstBaron'] == 0]

fig = go.Figure(data=[
    go.Pie(
        labels=['Blue Win with First Baron', 'Blue Win without First Baron'],
        values=[np.sum(blue_win_fbaron['blueWins']), np.sum(blue_win_nbaron['blueWins'])]
    )
])

fig.update_layout(
    title='Blue Wins With and Without First Baron',
    height=800,
    width=800
)
fig = go.Figure(data=[
    go.Histogram(
        name='Blue Team Killed by Towers (Won)',
        x=blue_win['red_towerKills']
    ),
    go.Histogram(
        name='Blue Team Killed by Towers (Lost)',
        x=red_win['red_towerKills']
    ),
    go.Histogram(
        name='Red Team Killed by Towers (Won)',
        x=red_win['blue_towerKills']
    ),
    go.Histogram(
        name='Red Team Killed by Towers (Lost)',
        x=blue_win['blue_towerKills']
    )
])

fig.update_layout(
    title='Distribution of Deaths due to Towers',
    height=800,
    width=800
)

iplot(fig)
import matplotlib.style as style

df_for_corr = df.copy()

plt.figure(figsize=(20, 20))

style.use('seaborn-poster')

df_for_corr.drop(bans + champs + ['redWins', 'redFirstBlood', 'red_firstInhibitor', 'red_firstBaron', 'red_firstRiftHerald'],  axis=1, inplace=True)
corr_df = df_for_corr.corr()

sns.heatmap(corr_df)
plt.title("Correlation Matrix", fontsize=25)
plt.tight_layout()
def evaluate_dist(df, cols):
    champ_cols = ['blue_champ_1', 'blue_champ_2', 'blue_champ_3', 'blue_champ_4', 'blue_champ_5',
                  'red_champ_1', 'red_champ_2', 'red_champ_3', 'red_champ_4', 'red_champ_5', 'ban_1',
                  'ban_2', 'ban_3', 'ban_4', 'ban_5', 'ban_6', 'ban_7', 'ban_8', 'ban_9', 'ban_10']
    kurtosis_results = dict()
    skewness = dict()

    for i in cols:
        if i not in champ_cols:
            kurtosis_results[i] = kurtosis(df[i])
            skewness[i] = skew(df[i])
    
    kurtosis_results = {k: v for k, v in sorted(kurtosis_results.items(), key=lambda item: item[1])}
    skewness = {k: v for k, v in sorted(skewness.items(), key=lambda item: item[1])}

    print('Skewness:\n')
    [print("{}: {}".format(i, skewness[i])) for i in skewness]

    print('\nKurtosis:\n')
    [print("{}: {}".format(i, kurtosis_results[i])) for i in kurtosis_results]

    print("\n\n")
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from prince import MCA
numerical_cols = [i for i in df if df[i].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']]

evaluate_dist(preproc_df, numerical_cols)
cols_to_be_transformed = ['blueWardsDestroyed', 'redWardsDestroyed',
                          'blueWardsPlaced', 'redWardsPlaced',
                          'redTowersDestroyed', 'blueTowersDestroyed']

for col in cols_to_be_transformed:
    preproc_df[col] = np.log1p(preproc_df[col])

evaluate_dist(preproc_df, cols_to_be_transformed)
combined_df.drop(['redFirstBlood', 'red_firstInhibitor', 'red_firstBaron', 'red_firstRiftHerald', 'gameId'], axis=1, inplace=True)

train_target = combined_df['blueWins'].iloc[:len(preproc_df)].reset_index(drop=True)
test_target = combined_df['blueWins'].iloc[len(preproc_df):].reset_index(drop=True)

champ_cols = ['blue_champ_1', 'blue_champ_2', 'blue_champ_3', 'blue_champ_4', 'blue_champ_5',
                  'red_champ_1', 'red_champ_2', 'red_champ_3', 'red_champ_4', 'red_champ_5', 'ban_1',
                  'ban_2', 'ban_3', 'ban_4', 'ban_5', 'ban_6', 'ban_7', 'ban_8', 'ban_9', 'ban_10']

combined_df.drop(['blueWins', 'redWins'], axis=1, inplace=True)

for col in champ_cols:
    combined_df[col] = get_champions(list(combined_df[col].values))

numerical_cols = [i for i in combined_df if combined_df[i].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']]



cols_to_be_transformed = ['blueWardsDestroyed', 'redWardsDestroyed',
                          'blueWardsPlaced', 'redWardsPlaced',
                          'redTowersDestroyed', 'blueTowersDestroyed']

# Split the combined dfs back to the original data set and my own test data set
train_df = combined_df.iloc[:len(preproc_df)].reset_index(drop=True)
test_df = combined_df.iloc[len(preproc_df):, :].reset_index(drop=True)


# Preprocess train data

df_for_scale = train_df[train_df.columns[~train_df.columns.isin(champ_cols)]]

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_for_scale)
pca = PCA(.95)
pcs = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pcs, columns=['PC_{}'.format(i) for i in range(np.size(pcs, 1))])

champ_df = train_df[train_df.columns[train_df.columns.isin(champ_cols)]]
champ_select_df = champ_df[champ_cols[:10]]
champ_ban_df = champ_df[champ_cols[10:]]

mca_ban = MCA(n_components=5)
mca_select = MCA(n_components=3)

ban_mca = mca_ban.fit_transform(champ_ban_df)
select_mca = mca_select.fit_transform(champ_select_df)

ban_mca.columns = ['MCA_Ban_{}'.format(i) for i in range(np.size(ban_mca, 1))]
select_mca.columns = ['MCA_Select_{}'.format(i) for i in range(np.size(select_mca, 1))]

train_reduced_df = pd.concat([ban_mca, select_mca, pca_df], axis=1)

# Preprocess Test Data

test_df_for_scale = test_df[test_df.columns[~test_df.columns.isin(champ_cols)]]

scaled_data = scaler.transform(test_df_for_scale)

pcs = pca.transform(scaled_data)

test_pca_df = pd.DataFrame(pcs, columns=['PC_{}'.format(i) for i in range(np.size(pcs, 1))])

champ_df = test_df[test_df.columns[test_df.columns.isin(champ_cols)]]
champ_select_df = champ_df[champ_cols[:10]]
champ_ban_df = champ_df[champ_cols[10:]]

ban_mca = mca_ban.fit_transform(champ_ban_df)
select_mca = mca_select.fit_transform(champ_select_df)

ban_mca.columns = ['MCA_Ban_{}'.format(i) for i in range(np.size(ban_mca, 1))]
select_mca.columns = ['MCA_Select_{}'.format(i) for i in range(np.size(select_mca, 1))]

test_reduced_df = pd.concat([ban_mca, select_mca, test_pca_df], axis=1)
train_reduced_df.head()
test_reduced_df.head()
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
x_train, x_test, y_train, y_test = train_test_split(train_reduced_df, train_target)
model = LogisticRegression(C=0.5, fit_intercept=True, n_jobs=-1, penalty='l2')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)
model = XGBClassifier(learning_rate=0.09, n_estimators=500, n_jobs=-1)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)
model = LogisticRegression(C=0.5, fit_intercept=True, n_jobs=-1, penalty='l2')

model.fit(train_reduced_df, train_target)

y_pred = model.predict(test_reduced_df)
logit_matrix = confusion_matrix(test_target, y_pred)
accuracy_score(test_target, y_pred)
model = XGBClassifier(learning_rate=0.09, n_estimators=500, n_jobs=-1, max_depth=5)

model.fit(train_reduced_df, train_target)
y_pred = model.predict(test_reduced_df)
xgboost_matrix = confusion_matrix(test_target, y_pred)
accuracy_score(test_target, y_pred)
model = RandomForestClassifier(bootstrap=True, 
                               max_depth=5,
                               max_features='auto',
                               min_samples_leaf=4, 
                               min_samples_split=5, 
                               n_estimators=500, 
                               oob_score=False)

model.fit(train_reduced_df, train_target)
y_pred = model.predict(test_reduced_df)
rf_matrix = confusion_matrix(test_target, y_pred)
accuracy_score(test_target, y_pred)
model = SVC(C=0.5, degree=1, gamma='auto', kernel='rbf')

model.fit(train_reduced_df, train_target)
y_pred = model.predict(test_reduced_df)
svm_matrix = confusion_matrix(test_target, y_pred)
accuracy_score(test_target, y_pred)
model_names = ["Logistic Regression", "XGBoost", "Random Forest", "SVM"]
model_results = [logit_matrix, xgboost_matrix, rf_matrix, svm_matrix]

fig, axs = plt.subplots(2,2, figsize=(11, 8))
grid_counter = 0
for i in range(2):
    for j in range(2):
        sns.heatmap(model_results[grid_counter], cmap='Blues', cbar=False, annot=True,
                    fmt='g', annot_kws={'size': 14}, ax=axs[i][j])

        axs[i][j].title.set_text(model_names[grid_counter])

        grid_counter += 1
plt.tight_layout()
plt.show()