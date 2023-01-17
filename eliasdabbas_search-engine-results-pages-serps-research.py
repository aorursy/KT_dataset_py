!pip install "advertools==0.7.4"
import pandas as pd

(pd.read_csv('../input/pymarketing.csv').head())
cx = 'YOUR_CUSTOM_SEARCH_KEY'

key = 'YOUR_GOOGLE_DEV_KEY'



import advertools as adv

import pandas as pd

pd.options.display.max_columns = None

import plotly

import plotly.graph_objs as go



for p in [adv, pd, plotly]:

    print(p.__name__, p.__version__)
yoga_mats = pd.read_csv('../input/yoga_mats.csv', parse_dates=['queryTime'])

print(yoga_mats.shape)

yoga_mats.head(3)
yoga_mats_multi = pd.read_csv('../input/yoga_mats_multi.csv', parse_dates=['queryTime'])

print(yoga_mats_multi.shape)

yoga_mats_multi.groupby('searchTerms').head(2)
import inspect

print(*inspect.signature(adv.serp_goog).parameters.keys(), sep='\n')
adv.SERP_GOOG_VALID_VALS.keys()
adv.SERP_GOOG_VALID_VALS['imgSize']
adv.SERP_GOOG_VALID_VALS['rights']
make_model = ['Chevrolet Malibu','Hyundai Sonata','Ford Escape',

              'Hyundai Elantra','Kia Sportage','Nissan Sentra',

              'Hyundai Santa Fe Sport','Ford Fusion','Nissan Altima',

              'Nissan Rogue','GMC Terrain','Kia Sorento','Toyota Camry',

              'Volkswagen Passat','Kia Forte','Chevrolet Traverse',

              'Ford Mustang','Dodge Dart','Ford Focus','Chrysler 200',

              'Ford Explorer','Toyota Corolla','Mitsubishi Lancer',

              'Nissan Versa','Kia Sedona','Toyota Prius','Nissan Versa Note',

              'Buick Enclave','Jeep Patriot','Toyota RAV4','Chevrolet Tahoe',

              'Nissan Pathfinder','Toyota Yaris','Jeep Grand Cherokee',

              'Dodge Charger','Ford Edge','Jeep Compass','Nissan Frontier',

              'Hyundai Santa Fe','Chevrolet Malibu Limited','Nissan JUKE',

              'Volkswagen Beetle Coupe','Jeep Cherokee','Ford Fiesta',

              'INFINITI QX60','Ram 1500','INFINITI QX70','Hyundai Accent',

              'Buick Regal','Dodge Durango'

]
q_for_sale = [x + ' for sale' for x in make_model]

q_price = [x + ' price' for x in make_model]

queries = [q.lower() for q in q_for_sale + q_price]

print('Number of queries: 50 (make-model combinations) x 2 (keyword variations) x 2 (countries) = 200 queries\nSample:')

queries[:5] + queries[-5:]
# serp_cars = adv.serp_goog(q=queries, cx=cx, key=key, gl=['us', 'uk'])
cars4sale  = pd.read_csv('../input/cars_forsale_price_us_uk.csv', parse_dates=['queryTime'])
print(cars4sale.shape)

cars4sale.groupby(['searchTerms']).head(2)[:10]
(cars4sale[cars4sale['gl'] == 'us']

 .pivot_table('rank', 'displayLink', aggfunc=['mean', 'count'])

 .sort_values([('count', 'rank')], ascending=False)

 .assign(cumsum=lambda df: df[('count', 'rank')].cumsum(),

         cum_perc=lambda df: df['cumsum'].div(df[('count', 'rank')].sum()))

 .head(10)

 .style.format({('cum_perc',''): '{:.2%}', ('mean', 'rank'): '{:.1f}'})

 .set_caption('Top domains in USA'))
top10domains_us = (cars4sale[cars4sale['gl'] == 'us']

                   ['displayLink'].value_counts().index[:10])

top10_df = (cars4sale[(cars4sale['gl'] == 'us') & 

                      (cars4sale['displayLink'].isin(top10domains_us))])

print(top10_df.shape)

top10_df.head(2)
rank_counts = (top10_df

               .groupby(['displayLink', 'rank'])

               .agg({'rank': ['count']})

               .reset_index())

rank_counts.columns = ['displayLink', 'rank', 'count']

rank_counts.head()
from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

fig = go.Figure()

fig.add_scatter(x=top10_df['displayLink'].str.replace('www.', ''),

                y=top10_df['rank'], mode='markers',

                marker={'size': 30, 

                        'opacity': 1/rank_counts['count'].max()})

fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', ''),

                y=rank_counts['rank'], mode='text',

                text=rank_counts['count'])



for domain in rank_counts['displayLink'].unique():

    rank_counts_subset = rank_counts[rank_counts['displayLink'] == domain]

    fig.add_scatter(x=[domain.replace('www.', '')],

                    y=[0], mode='text',

                    marker={'size': 50},

                    text=str(rank_counts_subset['count'].sum()))



    fig.add_scatter(x=[domain.replace('www.', '')],

                    y=[-1], mode='text',

                    text=format(rank_counts_subset['count'].sum() / top10_df['queryTime'].nunique(), '.1%'))

    fig.add_scatter(x=[domain.replace('www.', '')],

                    y=[-2], mode='text',

                    marker={'size': 50},

                    text=str(round(rank_counts_subset['rank']

                                   .mul(rank_counts_subset['count'])

                                   .sum() / rank_counts_subset['count']

                                   .sum(), 2)))



minrank, maxrank = min(top10_df['rank'].unique()), max(top10_df['rank'].unique())

fig.layout.yaxis.tickvals = [-2, -1, 0] + list(range(minrank, maxrank+1))

fig.layout.yaxis.ticktext = ['Avg. Pos.', 'Coverage', 'Total<br>appearances'] + list(range(minrank, maxrank+1))



fig.layout.height = 600 #max([600, 100 + ((maxrank - minrank) * 50)])

fig.layout.width = 1000

fig.layout.yaxis.title = 'SERP Rank (number of appearances)'

fig.layout.showlegend = False

fig.layout.paper_bgcolor = '#eeeeee'

fig.layout.plot_bgcolor = '#eeeeee'

fig.layout.autosize = False

fig.layout.margin.r = 2

fig.layout.margin.l = 120

fig.layout.margin.pad = 0

fig.layout.hovermode = False

fig.layout.yaxis.autorange = 'reversed'

fig.layout.yaxis.zeroline = False

fig.layout.template = 'none'

fig.layout.title = 'Top domains ranking for used car keywords in the US'

iplot(fig)
(cars4sale

 [(cars4sale['displayLink'].isin(['www.cargurus.com', 'www.truecar.com', 'www.edmunds.com'])) & 

  (cars4sale['title'].str.contains('Ford Escape'))][['title','link']])
for position in [41, 45, 46, 50, 55, 1044, 1048, 1049, 1054]:

    print(cars4sale['searchTerms'][position])

    print('='*23)

    print(cars4sale['title'][position])

    print(cars4sale['link'][position])

    print('-' * 23, '\n')
print(cars4sale.filter(regex='og:').shape)

cars4sale.filter(regex='og:').head()
print(cars4sale.filter(regex='twitter:').shape)

cars4sale.filter(regex='twitter:').dropna(how='all').sample(3)
print(cars4sale.filter(regex='al:').shape)

cars4sale.filter(regex='al:').dropna(how='all')
cars4sale.filter(regex='rating').dropna(how='all').sample(4)
# basketball = adv.serp_goog(q="basketball", cx=cx, key=key, start=[1, 11, 21])
basketball = pd.read_csv('../input/basketball.csv', parse_dates=['queryTime'])

print('Rows, columns:', basketball.shape)

print('\nRanks:', basketball['rank'].values, '\n')

basketball.head(2)
# trump = adv.serp_goog(q='donald trump', cx=cx, key=key, dateRestrict=['d1', 'd30','m6'])
trump = pd.read_csv('../input/trump.csv', parse_dates=['queryTime'])

trump.groupby(['dateRestrict']).head()[['rank','dateRestrict','title', 'snippet', 'displayLink', 'queryTime']]