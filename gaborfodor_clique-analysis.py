
%matplotlib inline
import os
import pandas as pd
pd.options.display.max_columns = 99
pd.options.display.max_rows = 100
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams['font.size'] = 16
start = dt.datetime.now()
clique_members = ['rose_brand', 'thrice_upon_a_matter', 'the_more_you_fail','round_ball',
                  'sellbuyhold', 'twice_upon_a_space', 'and_efficacy', 'transparency',
                  'the_guy', 'the_more_you_cry', 'the_more_you_try', 'dvb', 'dvc',
                  'once_upon_a_time', 'balling_the_round', 'we_were_young', 'around_the_ball',
                  'who_made_who', 'silver_medal', 'rose_new', 'is_this_the_life',
                  'the_last_refugee', 'rose_old', 'ataulfo1', 'buyholdsell', 'ataulfo2',
                  'tommy_who', 'madmax', 'buysellhold', 'dva','ataulfo3', 'whos_next']
T = 110
FIRST_STAKE_ROUND = 61
ROUNDS = range(FIRST_STAKE_ROUND, T + 1)
STAKING_ROUNDS = range(FIRST_STAKE_ROUND, T + 1)
lb_with_stakes = pd.read_csv('../input/leaderboards.csv')
lb_with_stakes['stake.insertedAt'] = pd.to_datetime(lb_with_stakes['stake.insertedAt'], errors='coerce', utc=True)
lb_with_stakes['overfitting'] = lb_with_stakes['liveLogloss'] - lb_with_stakes['validationLogloss']
lb_with_stakes.head()
lb_with_stakes.shape
possible_matches = pd.read_csv('../input/possible_matches.csv')
possible_matches.head()
possible_matches.shape
clique_possible_matches = possible_matches[possible_matches.user_name_x.isin(clique_members)]
clique_possible_matches = clique_possible_matches[clique_possible_matches.user_name_y.isin(clique_members)]
user_coords = pd.read_csv('../input/T{}_user_coords_features.csv'.format(T))
user_coords = user_coords[user_coords.user_name.isin(clique_members)]
user_coords = user_coords.set_index('user_name')
data = []
for u1, u2 in clique_possible_matches[['user_name_x', 'user_name_y']].values:
    trace = go.Scatter(
        x = [user_coords.loc[u1, 'x'], user_coords.loc[u2, 'x']],
        y = [user_coords.loc[u1, 'y'], user_coords.loc[u2, 'y']],
        mode = 'lines',
        line=dict(color='grey', width=1))
    data.append(trace)
data.append(
    go.Scatter(
        y = user_coords['y'],
        x = user_coords['x'],
        mode='markers+text',
        marker=dict(sizemode='diameter',sizeref=1, size=10, color='black'),
        text=user_coords.index,
        hoverinfo = 'text',
        textposition=["top center"],
    )
)
layout = go.Layout(
    autosize=True,
    title='User similarity',
    hovermode='closest',
    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    xaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='user_similarity for high degree vertices')
user_candidate_file_name = '../input/T{}_user_candidate_features.csv'.format(T)
user_candidates = pd.read_csv(user_candidate_file_name)
user_candidates.head()

clique_possible_matches = user_candidates[user_candidates.user_name_x.isin(clique_members)]
clique_possible_matches = clique_possible_matches[clique_possible_matches.user_name_y.isin(clique_members)]
clique_possible_matches.to_csv('clique_possible_matches.csv', index=False)
clique_possible_matches.shape
user_candidates.shape
clique_possible_matches.describe()
user_candidates.describe()
user_candidates.columns
for col in ['l1_rank_diff', 'corr_vll', 'median_stake_time_diff']:
    fig, ax = plt.subplots()
    plt.hist(user_candidates[col].values, bins=25, density=True, alpha=0.5, label='all')
    plt.hist(clique_possible_matches[col].values, bins=25, density=True, alpha=0.5, label='clique')
    plt.grid()
    plt.title(col)
    plt.xlabel(col)
    plt.legend(loc=0)
plt.show();
clique_closed_lb_results = lb_with_stakes[lb_with_stakes.username.isin(clique_members)]
clique_closed_lb_results = clique_closed_lb_results[clique_closed_lb_results.number <= 107]
clique_closed_lb_results.shape
clique_closed_lb_results.head()
'Live consistency: {:.1f}%'.format(100 * clique_closed_lb_results.better_than_threshold.mean())
'Staking payments: {:.0f} NMR and {:.0f} USD'.format(clique_closed_lb_results['paymentStaking.nmrAmount'].sum(),
                                                     clique_closed_lb_results['paymentStaking.usdAmount'].sum())
end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))