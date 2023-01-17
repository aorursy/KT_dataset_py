# %matplotlib inline

import pandas as pd

import datetime as dt

import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import IPython.display as display

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

import seaborn as sns

sns.set_palette(sns.color_palette('tab20', 20))

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff
start = dt.datetime.now()

T = 166

FIRST_ROUND = 94

ROUNDS = range(FIRST_ROUND, T + 1)

STAKING_ROUNDS = range(FIRST_ROUND, T + 1)

THRESHOLD = 0.693
nmr_price = pd.read_csv('../input/CoinMarketCapNMR')

nmr_price['Date'] = [dt.datetime.strptime(d, '%b-%d-%Y') for d in nmr_price.Date.values]

nmr_price['Date'] = [pd.Timestamp(d).date() for d in nmr_price.Date.values]

nmr_price = nmr_price[['Date', 'High']]

nmr_price.columns = ['Date', 'NMRUSD']

nmr_price = nmr_price[nmr_price.Date >= dt.date(2018, 2, 1)]
data = []

trace = go.Scatter(

    x = nmr_price.Date.values,

    y = nmr_price.NMRUSD.values,

    mode = 'lines',

    name = 'NMR - USD',

    line=dict(width=4)

)

data.append(trace)

layout= go.Layout(

    title= 'NMR historical price',

    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),

    yaxis=dict(title='NMR USD', ticklen=5, gridwidth=2),

    showlegend=True

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='nmr_usd')
competitions = pd.read_csv('../input/T{}_competitions.csv'.format(T), low_memory=False)

lb_with_stakes = pd.read_csv('../input/T{}_leaderboards.csv'.format(T), low_memory=False)

lb_with_stakes = lb_with_stakes[lb_with_stakes['number'] >= FIRST_ROUND]
tournament_results = lb_with_stakes.drop([

    'tournament_id', 'consistency', 'concordance.value', 'better_than_random', 'stake.insertedAt', 'stakeResolution.paid',

], axis=1)

tournament_results = tournament_results.merge(competitions, how='left', on=['number'])

tournament_results = tournament_results.drop([

    'datasetId', 'participants', 'prizePoolNmr', 'prizePoolUsd'], axis=1)

tournament_results['ResolveDate'] = [pd.Timestamp(d).date() for d in tournament_results.resolveTime.values]

tournament_results['OpenDate'] = [pd.Timestamp(d).date() for d in tournament_results.openTime.values]

tournament_results = tournament_results.merge(nmr_price, how='left', left_on='ResolveDate', right_on='Date')

tournament_results = tournament_results.merge(nmr_price, how='left', left_on='OpenDate', right_on='Date',

                                              suffixes=['Resolve', 'Open'])

tournament_results = tournament_results.drop([

    'openTime', 'resolveTime', 'DateResolve', 'DateOpen', 'resolvedGeneral', 'resolvedStaking'], axis=1)

resolved = tournament_results[~tournament_results.liveLogloss.isna()]
tour_performance_median = resolved.groupby('number').median()

fig, ax = plt.subplots()

plt.plot(tour_performance_median.index, tour_performance_median.liveLogloss, c='r', lw=5, label='medianliveLogloss')

plt.plot(tour_performance_median.index, tour_performance_median.validationLogloss, c='b', lw=5, label='medianvalidationLogloss')

plt.plot(tour_performance_median.index, [np.log(2)] * tour_performance_median.shape[0], 'k:', lw=2, label='ln(2)')

plt.plot(tour_performance_median.index, [THRESHOLD] * tour_performance_median.shape[0], 'k-', lw=2, label='Threshold: 0.693')

plt.scatter(resolved.number.values, resolved.liveLogloss.values, color='g', s=20, alpha=0.2, label='liveLogloss')

plt.grid()

plt.title('Live Logloss Results')

plt.xlabel('Round')

plt.ylabel('Logloss')

plt.legend(loc=0)

plt.ylim(0.69, 0.7)

plt.show();
users_with_submission = tournament_results.groupby('OpenDate')[['username']].nunique()

users_with_stake = tournament_results[~tournament_results['stake.value'].isna()].groupby('OpenDate')[['username']].nunique()

user_counts = pd.merge(users_with_submission, users_with_stake, left_index=True, right_index=True).reset_index()

user_counts.columns = ['OpenDate', 'UserswithSubmission', 'UserswithStake']
data = [

go.Scatter(

    x = user_counts.OpenDate.values,

    y = user_counts.UserswithSubmission.values,

    mode = 'lines',

    name = '#Users with submission',

    line=dict(width=4)

),

go.Scatter(

    x = user_counts.OpenDate.values,

    y = user_counts.UserswithStake.values,

    mode = 'lines',

    name = '#Users with stake',

    line=dict(width=4)

),

]

layout= go.Layout(

    title= 'User stats',

    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),

    yaxis=dict(title='Number of users', ticklen=5, gridwidth=2),

    showlegend=True

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='users')
total_stake_amount = tournament_results.groupby('OpenDate')[['stake.value']].sum().reset_index()

data = [

go.Bar(

    x = total_stake_amount.OpenDate.values,

    y = total_stake_amount['stake.value'].values,

    name = 'Stake',

),

]

layout= go.Layout(

    title= '"Skin in the game"',

    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),

    yaxis=dict(title='Total stakes (NMR)', ticklen=5, gridwidth=2),

    showlegend=True

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='skin')
stakes = resolved[~resolved['stake.value'].isna()]

stakes = stakes.fillna(0)

stakes['Burned'] = 1.0 * stakes['stakeResolution.destroyed'] * stakes['stake.value']

stakes['Winning'] = 1.0 * stakes['stakeResolution.successful'] * stakes['stake.value']

stakes['Unused'] = stakes['stake.value'] - stakes['Winning'] - stakes['Burned']

stake_type = stakes.groupby('OpenDate')[['Burned', 'Winning', 'Unused']].sum().reset_index()
total_stake_amount = tournament_results.groupby('OpenDate')[['stake.value']].sum().reset_index()

data = [

    go.Bar(x=stake_type.OpenDate.values, y=stake_type['Burned'].values,

           marker=dict(color='red'), name = 'Burn',),

    go.Bar(x=stake_type.OpenDate.values, y=stake_type['Winning'].values,

           marker=dict(color='green'), name = 'Winning',),

    go.Bar(x=stake_type.OpenDate.values, y=stake_type['Unused'].values,

           marker=dict(color='grey'), name = 'Unused',),

]

layout= go.Layout(

    barmode='stack',

    title= 'Stakes',

    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),

    yaxis=dict(title='Stake (NMR)', ticklen=5, gridwidth=2),

    showlegend=True

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stake-type')
stakes['PrizeUSD'] = stakes['paymentStaking.nmrAmount'] * stakes['NMRUSDResolve'] + stakes['paymentStaking.usdAmount']

stakes['PrizeUSD5'] = stakes['paymentStaking.nmrAmount'] * 5 + stakes['paymentStaking.usdAmount']

stakes['ProfitUSD'] = stakes['PrizeUSD'] - stakes['Burned'] * stakes['NMRUSDOpen'] + stakes['Winning'] * (stakes['NMRUSDResolve'] - stakes['NMRUSDOpen'])

stakes = stakes.fillna(0)

PrizeUSD = stakes.groupby('OpenDate')[['PrizeUSD', 'ProfitUSD', 'PrizeUSD5']].sum().reset_index()
data = [

go.Bar(

    x = PrizeUSD.OpenDate.values,

    y = PrizeUSD['PrizeUSD'].values,

    name = 'PrizeUSD',

)]

layout= go.Layout(

    title= 'Actual prize pool',

    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),

    yaxis=dict(title='Paid Prize (USD)', ticklen=5, gridwidth=2),

    showlegend=True

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='prize')
data = [

go.Bar(

    x = PrizeUSD.OpenDate.values,

    y = PrizeUSD['ProfitUSD'].values,

    name = 'ProfitUSD',

)]

layout= go.Layout(

    title= 'Profit',

    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),

    yaxis=dict(title='Profit (USD)', ticklen=5, gridwidth=2),

    showlegend=True

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='profit')
nt = stakes.groupby(['number', 'tournament_name'])[['better_than_threshold']].mean().reset_index()

ntp = nt.pivot('number', 'tournament_name', 'better_than_threshold')

ntp = ntp[ntp.index >= 110]
data = [

go.Scatter(x = ntp.index.values, y = ntp[tname].values, mode='lines+markers',

           marker=dict(size=10), name=tname, line=dict(width=2), opacity=0.7)

    for tname in ntp.columns

]

layout= go.Layout(

    title= 'Live Success Rate',

    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),

    yaxis=dict(title='P(better than threshold)', ticklen=5, gridwidth=2),

    showlegend=True

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='consistency')
end = dt.datetime.now()

print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))