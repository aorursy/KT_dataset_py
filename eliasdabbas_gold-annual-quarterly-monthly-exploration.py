import pandas as pd

import plotly

import plotly.graph_objects as go
gold = pd.read_csv('../input/gold-reserves-by-country-quarterly/gold_reserves_annual_quarterly_monthly.csv')

gold
russia_monthly = gold[(gold['Country Name']=='Russian Federation') & (gold['period'] == 'month')]

russia_monthly
turkey_quarterly = gold[(gold['Country Name']=='Turkey') & (gold['period'] == 'quarter')]

turkey_quarterly
gold['Country Name'][gold['Country Name'].str.contains('china', case=False)].unique()
china_annual = gold[(gold['Country Name']=='China, P.R.: Mainland') & (gold['period'] == 'year')]

china_annual.head()
q209_q219 = gold[gold['Time Period'].isin(['2009Q2', '2019Q2'])]

q209_q219
print('Top 10 Countries that changed gold reserves 2019Q2 vs. 2009Q1')

pivoted = (q209_q219

           .pivot_table(index='Country Name', columns='Time Period', values='tonnes')

           .assign(diff=lambda df: df['2019Q2']-df['2009Q2'])

           .dropna()

           .sort_values('diff', ascending=False))

q2_top10 = pivoted.head(10).append(pivoted.tail(10))



q2_top10.style.format("{:,.0f}")
fig = go.Figure()

fig.add_bar(y=q2_top10.index[::-1], x=q2_top10['diff'][::-1], 

            orientation='h', marker={'color': (['red'] * 10) + (['green']*10)})

fig.layout.height = 600

fig.layout.xaxis.title = 'Tonnes'

fig.layout.title = 'Top Gainers and Losers of Gold Reserves 2019-Q2 vs 2009-Q2'



fig
def top_gainers_losers(df=gold, first_quarter='2009Q2', second_quarter='2019Q2', top_n=10):

    df = df[df['Time Period'].isin([first_quarter, second_quarter])]

    pivoted = (df

               .pivot_table(index='Country Name', columns='Time Period', values='tonnes')

               .assign(diff=lambda df: df.iloc[:, 1] - df.iloc[:, 0])

               .dropna()

               .sort_values('diff', ascending=False))

    top10 = pivoted.head(top_n).append(pivoted.tail(top_n))

    fig = go.Figure()

    fig.add_bar(y=top10.index[::-1], x=top10['diff'][::-1], 

                orientation='h', marker={'color': (['red'] * 10) + (['green']*10)})

    fig.layout.height = 600

    fig.layout.xaxis.title = 'Tonnes'

    fig.layout.title = 'Top Gainers and Losers of Gold Reserves ' + second_quarter + ' vs. ' + first_quarter



    return fig
top_gainers_losers(df=gold, first_quarter='2005Q2', second_quarter='2007Q1')
to_remove = ['Europe', 'CIS', 'Middle East, North Africa, Afghanistan, and Pakistan', 'Sub-Saharan Africa',

             'Emerging and Developing Asia', 'Euro Area', 'Advanced Economies', 'World']
top_gainers_losers(df=gold[~gold['Country Name'].isin(to_remove)], first_quarter='2005Q2', second_quarter='2007Q1')
top_gainers_losers(df=gold[~gold['Country Name'].isin(to_remove)], first_quarter='2000Q1', second_quarter='2010Q1')
top_gainers_losers(df=gold[~gold['Country Name'].isin(to_remove)], first_quarter='1990Q1', second_quarter='2000Q1')