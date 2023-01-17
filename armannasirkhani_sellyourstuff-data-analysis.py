import numpy as np

import pandas as pd

import sqlite3

import itertools

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import plotly.graph_objs as go

from datetime import datetime

from dateutil.relativedelta import relativedelta
conn = sqlite3.connect('../input/sellyourstuff/company.db')

Campaigns = pd.read_sql('SELECT * FROM Campaigns', con = conn)

Clients = pd.read_sql('SELECT account, type, residence, date_joined, indication_coupon, first_deposit_amount, first_deposit_date, first_transaction_amount, first_transaction_date, balance_amount FROM Clients', con = conn)

Transactions = pd.read_sql('SELECT transaction_date, account, total_buy, total_sell, total_deposits, total_withdrawals, count_contracts, count_deposits, count_withdrawals FROM Transactions', con = conn)



country_codes = pd.read_csv('../input/country-codes/wikipedia-iso-country-codes.csv', usecols=['Alpha-2 code', 'Alpha-3 code'])

for country in filter(None, Campaigns['country'].unique()):

    Campaigns['country'] = Campaigns['country'].replace([country], country_codes['Alpha-3 code'].loc[country_codes['Alpha-2 code'] == country.upper()], regex=True)



Clients = Clients.rename({'residence': 'country'}, axis=1)

Clients['country'] = Clients['country'].replace(['na'], 'NAM', regex=True)

for country in filter(None, Clients['country'].unique()):

    if country != 'NAM':

        Clients['country'] = Clients['country'].replace([country], country_codes['Alpha-3 code'].loc[country_codes['Alpha-2 code'] == country.upper()], regex=True)



Transactions = Transactions.assign(country=[None]*len(Transactions))

for account in Clients['account']:

    ii = np.where(Transactions['account'] == account)[0]

    Transactions.loc[ii, ['country']] = Clients['country'].loc[Clients['account'] == account].iloc[0]



Campaigns=Campaigns.replace(r'^\s*$', 'global', regex=True)



Campaigns[['start_date', 'end_date']] = Campaigns[['start_date', 'end_date']].apply(pd.to_datetime)

Clients[['date_joined', 'first_deposit_date', 'first_transaction_date']] = Clients[['date_joined', 'first_deposit_date', 'first_transaction_date']].apply(pd.to_datetime)

Clients[['account', 'indication_coupon']] = Clients[['account', 'indication_coupon']].astype(object)

Transactions['account'] = Transactions['account'].astype(object)

Transactions['transaction_date'] = Transactions['transaction_date'].astype('datetime64')

Campaigns.head()
pd.DataFrame(Campaigns['country'].value_counts())
Clients.head()
temp = pd.DataFrame(Clients['country'].value_counts())

fig=go.Figure(data=go.Choropleth(

            locations = temp.index,

            z = Clients['country'].value_counts(),

            autocolorscale=True,

            reversescale=False,

            marker_line_color='darkgray',

            marker_line_width=0.5))

fig.show()
Clients[['type', 'indication_coupon']].describe(include=[object])
Transactions.head()
start_date = datetime(2017, 1, 1)

end_date = datetime(2018, 11, 1)



time_periods = []

while start_date <= end_date:

    time_periods.append(start_date)

    start_date += relativedelta(months=2)
def worldmaptimed(table, timecol, statcol, time_stamps, country_codes, instance_name, exclude, plot_flag):

    country_stat_per_division = pd.DataFrame(columns=[item.strftime("%Y/%m") for item in time_stamps])

    country_stat_per_division = country_stat_per_division.rename({time_stamps[0].strftime("%Y/%m"):'country'}, axis=1)

    country_stat_per_division['country'] = country_codes['Alpha-3 code']

    fig = go.Figure()

    for i in range(len(time_stamps)-1):

        df_cumsum_month = pd.DataFrame({'country':[], 'stat':[], 'count':[]})

        if table[statcol].loc[(table[timecol] >= time_stamps[i]) & (table[timecol] < time_stamps[i+1]) & (table['country'] == 'global')].any():

            df_cumsum_month['country'] = country_codes['Alpha-3 code']

            df_cumsum_month['stat'] = table[statcol].loc[(table[timecol] >= time_stamps[i]) & (table[timecol] < time_stamps[i+1]) & (table['country'] == 'global')].sum()/len(country_codes)

            df_cumsum_month['count'] = table[statcol].loc[(table[timecol] >= time_stamps[i]) & (table[timecol] < time_stamps[i+1]) & (table['country'] == 'global')].count()

            country_stat_per_division[time_stamps[i+1].strftime("%Y/%m")] = df_cumsum_month['stat']

            for country in table['country'].loc[(table[timecol] >= time_stamps[i]) & (table[timecol] < time_stamps[i+1])].unique():

                if (country != 'global') & (country != exclude):

                    data = Campaigns.loc[(table[timecol] >= time_stamps[i]) & (table[timecol] < time_stamps[i+1]) & (table['country'] == country)]

                    df_cumsum_month['stat'].loc[df_cumsum_month['country'] == country] += data[statcol].loc[data['country'] == country].sum()

                    df_cumsum_month['count'].loc[df_cumsum_month['country'] == country] += data[statcol].loc[data['country'] == country].count()

                    ii = np.where(country_stat_per_division['country'] == country)[0]

                    country_stat_per_division.loc[ii, time_stamps[i+1].strftime("%Y/%m")] = data[statcol].loc[data['country'] == country].sum() + table[statcol].loc[(table[timecol] >= time_stamps[i]) & (table[timecol] < time_stamps[i+1]) & (table['country'] == 'global')].sum()/len(country_codes)

        else:

            for country in table['country'].loc[(table[timecol] >= time_stamps[i]) & (table[timecol] < time_stamps[i+1])].unique():

                if country != exclude:

                    data = table.loc[(table[timecol] >= time_stamps[i]) & (table[timecol] < time_stamps[i+1]) & (table['country'] == country)]

                    df_cumsum_month = df_cumsum_month.append({'country': country, 'stat': data[statcol].loc[data['country'] == country].sum(),

                                                             'count': data[statcol].loc[data['country'] == country].count()}, ignore_index=True)

                    ii = np.where(country_stat_per_division['country'] == country)[0]

                    country_stat_per_division.loc[ii, time_stamps[i+1].strftime("%Y/%m")] = data[statcol].loc[data['country'] == country].sum()



        for col in df_cumsum_month.columns: 

            df_cumsum_month[col] = df_cumsum_month[col].astype(str)



        df_cumsum_month['text'] = 'number of '+ instance_name + ' = ' + df_cumsum_month['count']



        fig.add_trace(go.Choropleth(

            locations = df_cumsum_month['country'],

            z = df_cumsum_month['stat'],

            text = df_cumsum_month['text'],

            autocolorscale=True,

            reversescale=False,

            marker_line_color='darkgray',

            marker_line_width=0.5,

            colorbar_tickprefix = '$',

            colorbar_title = 'US$'))



    steps = []

    for i in range(len(fig.data)):

        step = dict(method='restyle',

                    args=['visible', [False] * len(fig.data)],

                    label=time_stamps[i+1].strftime("%Y/%m"))

        step['args'][1][i] = True

        steps.append(step)



    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]  



    fig.update_layout(

        title_text=instance_name,

        geo=dict(

            showframe=False,

            showcoastlines=False,

            projection_type='equirectangular'

        ),

        sliders=sliders

    )

    if plot_flag:

        fig.show()

    

    return country_stat_per_division
country_total_buy_per_division = worldmaptimed(Transactions, 'transaction_date', 'total_buy', time_periods, country_codes, 'transactions total_buy', 'IDN', 1)
country_campaign_spend_per_division = worldmaptimed(Campaigns, 'start_date', 'total_spend', time_periods, country_codes, 'campaigns', 'IDN', 0)

country_count_contracts_per_division = worldmaptimed(Transactions, 'transaction_date', 'count_contracts', time_periods, country_codes, 'transactions total_withdrawals', 'IDN', 0)

country_count_deposits_per_division = worldmaptimed(Transactions, 'transaction_date', 'count_deposits', time_periods, country_codes, 'transactions total_withdrawals', 'IDN', 0)

country_count_withdrawals_per_division = worldmaptimed(Transactions, 'transaction_date', 'count_withdrawals', time_periods, country_codes, 'transactions total_withdrawals', 'IDN', 0)

country_total_deposits_division = worldmaptimed(Transactions, 'transaction_date', 'total_deposits', time_periods, country_codes, 'transactions total_deposits', 'IDN', 0)

country_first_deposit_amount_per_division = worldmaptimed(Clients, 'first_deposit_date', 'first_deposit_amount', time_periods, country_codes, 'Clients first_deposit_amount', 'IDN', 0)

country_total_sell_per_division = worldmaptimed(Transactions, 'transaction_date', 'total_sell', time_periods, country_codes, 'transactions total_sell', 'IDN', 0)

country_total_withdrawals_per_division = worldmaptimed(Transactions, 'transaction_date', 'total_withdrawals', time_periods, country_codes, 'transactions total_withdrawals', 'IDN', 0)



a=pd.DataFrame(np.hstack(np.vsplit(country_total_buy_per_division.drop('country', axis=1),country_total_buy_per_division.shape[0]))).transpose()

b=pd.DataFrame(np.hstack(np.vsplit(country_total_sell_per_division.drop('country', axis=1),country_total_sell_per_division.shape[0]))).transpose()

c=pd.DataFrame(np.hstack(np.vsplit(country_total_withdrawals_per_division.drop('country', axis=1),country_total_withdrawals_per_division.shape[0]))).transpose()

d=pd.DataFrame(np.hstack(np.vsplit(country_total_deposits_division.drop('country', axis=1),country_total_deposits_division.shape[0]))).transpose()

e=pd.DataFrame(np.hstack(np.vsplit(country_first_deposit_amount_per_division.drop('country', axis=1),country_first_deposit_amount_per_division.shape[0]))).transpose()

f=pd.DataFrame(np.hstack(np.vsplit(country_count_contracts_per_division.drop('country', axis=1),country_count_contracts_per_division.shape[0]))).transpose()

g=pd.DataFrame(np.hstack(np.vsplit(country_count_deposits_per_division.drop('country', axis=1),country_count_deposits_per_division.shape[0]))).transpose()

h=pd.DataFrame(np.hstack(np.vsplit(country_count_withdrawals_per_division.drop('country', axis=1),country_count_withdrawals_per_division.shape[0]))).transpose()



data = pd.concat([a, b, c, d, e, f, g, h], axis=1).dropna(how='all').fillna(0)

data.columns = ['total_buy', 'total_sell', 'total_withdrawals', 'total_deposits', 'first_deposit_amount', 'count_contracts', 'count_deposits', 'count_withdrawals']

pca = PCA()

pca.fit(data)

print(pca.explained_variance_ratio_)

print(pca.components_)
# Initialize figure

fig = go.Figure()



# Add Traces

figure_matrix = country_total_buy_per_division.drop('country', axis=1).fillna(0) + country_total_deposits_division.drop('country', axis=1).fillna(0)

figure_matrix['total'] = figure_matrix.sum(axis=1)

figure_matrix['country'] = country_total_buy_per_division['country']

figure_matrix = figure_matrix.sort_values('total', ascending=0)

countries = figure_matrix['country']

for country in countries[:20]:

    fig.add_trace(

        go.Scatter(x=time_periods[1:],

                   y=figure_matrix.loc[figure_matrix['country'] == country].drop('country', axis=1).fillna(0).to_numpy()[0],

                   name='total prosper'))

    fig.add_trace(

        go.Scatter(x=time_periods[1:],

                       y=country_campaign_spend_per_division.loc[country_campaign_spend_per_division['country'] == country].drop('country', axis=1).to_numpy()[0],

                       name='campaign spend'))



fig.update_layout(

    updatemenus=[

        go.layout.Updatemenu(

            active=0,

            buttons = [dict(label=country,

                     method="update",

                     args=[{"visible": list(itertools.chain.from_iterable([itertools.repeat(item==country,2) for item in countries[:20]]))},

                           {"title": country}]) for country in countries[:20]],

        )

    ])



# Set title

fig.update_layout(title_text="country")

fig.show()
kmeans = KMeans(n_clusters=5, random_state=0).fit(figure_matrix[:20].drop(['country', 'total'], axis=1).fillna(0))

kmeans.labels_
fig2 = go.Figure()

fig2.add_trace(

    go.Scatter(x=time_periods[1:],

               y=figure_matrix.loc[figure_matrix['country'] == 'BRA'].drop('country', axis=1).fillna(0).to_numpy()[0],

               name='total prosper Brazil'))

fig2.add_trace(

    go.Scatter(x=time_periods[1:],

                   y=country_campaign_spend_per_division.loc[country_campaign_spend_per_division['country'] == 'BRA'].drop('country', axis=1).to_numpy()[0],

                   name='campaign spend Brazil'))

fig2.add_trace(

    go.Scatter(x=time_periods[1:],

               y=figure_matrix.loc[figure_matrix['country'] == 'NGA'].drop('country', axis=1).fillna(0).to_numpy()[0],

               name='total prosper Nigeria'))

fig2.add_trace(

    go.Scatter(x=time_periods[1:],

                   y=country_campaign_spend_per_division.loc[country_campaign_spend_per_division['country'] == 'NGA'].drop('country', axis=1).to_numpy()[0],

                   name='campaign spend Nigeria'))



# Set title

fig2.update_layout(title_text="Cluster 1")

fig2.show()
fig3 = go.Figure()

fig3.add_trace(

    go.Scatter(x=time_periods[1:],

               y=figure_matrix.loc[figure_matrix['country'] == 'GBR'].drop('country', axis=1).fillna(0).to_numpy()[0],

               name='total prosper England'))

fig3.add_trace(

    go.Scatter(x=time_periods[1:],

                   y=country_campaign_spend_per_division.loc[country_campaign_spend_per_division['country'] == 'GBR'].drop('country', axis=1).to_numpy()[0],

                   name='campaign spend England'))

fig3.add_trace(

    go.Scatter(x=time_periods[1:],

               y=figure_matrix.loc[figure_matrix['country'] == 'UKR'].drop('country', axis=1).fillna(0).to_numpy()[0],

               name='total prosper Ukraine'))

fig3.add_trace(

    go.Scatter(x=time_periods[1:],

                   y=country_campaign_spend_per_division.loc[country_campaign_spend_per_division['country'] == 'UKR'].drop('country', axis=1).to_numpy()[0],

                   name='campaign spend Ukraine'))



# Set title

fig3.update_layout(title_text="Cluster 2")

fig3.show()
fig4 = go.Figure()

fig4.add_trace(

    go.Scatter(x=time_periods[1:],

               y=figure_matrix.loc[figure_matrix['country'] == 'RUS'].drop('country', axis=1).fillna(0).to_numpy()[0],

               name='total prosper Russia'))

fig4.add_trace(

    go.Scatter(x=time_periods[1:],

                   y=country_campaign_spend_per_division.loc[country_campaign_spend_per_division['country'] == 'RUS'].drop('country', axis=1).to_numpy()[0],

                   name='campaign spend Russia'))

fig4.add_trace(

    go.Scatter(x=time_periods[1:],

               y=figure_matrix.loc[figure_matrix['country'] == 'COL'].drop('country', axis=1).fillna(0).to_numpy()[0],

               name='total prosper Colombia'))

fig4.add_trace(

    go.Scatter(x=time_periods[1:],

                   y=country_campaign_spend_per_division.loc[country_campaign_spend_per_division['country'] == 'COL'].drop('country', axis=1).to_numpy()[0],

                   name='campaign spend Colombia'))



# Set title

fig4.update_layout(title_text="Cluster 3")

fig4.show()
fig5 = go.Figure()

fig5.add_trace(

    go.Scatter(x=time_periods[1:],

               y=figure_matrix.loc[figure_matrix['country'] == 'VNM'].drop('country', axis=1).fillna(0).to_numpy()[0],

               name='total prosper Vietnam'))

fig5.add_trace(

    go.Scatter(x=time_periods[1:],

                   y=country_campaign_spend_per_division.loc[country_campaign_spend_per_division['country'] == 'VNM'].drop('country', axis=1).to_numpy()[0],

                   name='campaign spend Vietnam'))

# Set title

fig5.update_layout(title_text="Cluster 4")

fig5.show()
IDN_campaign_spend_per_division = worldmaptimed(Campaigns, 'start_date', 'total_spend', time_periods, country_codes, 'campaigns', '', 0)

IDN_total_deposits_division = worldmaptimed(Transactions, 'transaction_date', 'total_deposits', time_periods, country_codes, 'transactions total_deposits', '', 0)

IDN_total_buy_per_division = worldmaptimed(Transactions, 'transaction_date', 'total_buy', time_periods, country_codes, 'transactions total_buy', '', 0)

IDN_matrix = IDN_total_buy_per_division.drop('country', axis=1).fillna(0) + IDN_total_deposits_division.drop('country', axis=1).fillna(0)

IDN_matrix['total'] = IDN_matrix.sum(axis=1)

IDN_matrix['country'] = IDN_total_buy_per_division['country']

IDN_matrix = IDN_matrix.sort_values('total', ascending=0)

fig6 = go.Figure()

fig6.add_trace(

    go.Scatter(x=time_periods[1:],

               y=IDN_matrix.loc[IDN_matrix['country'] == 'IDN'].drop('country', axis=1).fillna(0).to_numpy()[0],

               name='total prosper'))

fig6.add_trace(

    go.Scatter(x=time_periods[1:],

                   y=IDN_campaign_spend_per_division.loc[IDN_campaign_spend_per_division['country'] == 'IDN'].drop('country', axis=1).to_numpy()[0],

                   name='campaign spend'))

# Set title

fig6.update_layout(title_text="Indonesia")

fig6.show()