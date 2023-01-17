import advertools as adv # package used to import data from Google's API

import pandas as pd # pandas is pandas! 

pd.options.display.max_columns = None

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



cx = 'YOUR_GOOGLE_CSE_SEARCH_ENGINE_ID' # get it here https://cse.google.com/cse/

key = 'YOUR_GOOGLE_DEVELOPER_KEY' # get it here https://console.cloud.google.com/apis/library/customsearch.googleapis.com

# coffee_df = adv.serp_goog(cx=cx, key=key, q='coffee',

#                           gl=adv.SERP_GOOG_VALID_VALS['gl'])



# cafe_df = adv.serp_goog(cx=cx, key=key, q='cafe',

#                         gl=adv.SERP_GOOG_VALID_VALS['gl'])
coffee_df = pd.read_csv('../input/coffee_serps.csv')

cafe_df = pd.read_csv('../input/cafe_serps.csv')



country_codes = sorted(adv.SERP_GOOG_VALID_VALS['gl'])

print('number of available locations:', len(country_codes))

print()

print('country codes:')

print(*[country_codes[x:x+15] for x in range(0, len(country_codes), 15)],

      sep='\n')
print('Coffee domains:', coffee_df['displayLink'].nunique())

print('Cafe domains:', cafe_df['displayLink'].nunique())

common = set(coffee_df['displayLink']).intersection(cafe_df['displayLink'])

print('# Common domains:', len(common))

common



num_domains = 15  # the number of domains to show in the chart

opacity = 0.02  # how opaque you want the circles to be

df = coffee_df  # which DataFrame you are using
top_domains = df['displayLink'].value_counts()[:num_domains].index.tolist()

top_domains
top_df = df[df['displayLink'].isin(top_domains)]

top_df.head(3)

top_df_counts_means = (top_df

                       .groupby('displayLink', as_index=False)

                       .agg({'rank': ['count', 'mean']})

                       .set_axis(['displayLink', 'rank_count', 'rank_mean'],

                                 axis=1, inplace=False))

top_df_counts_means

top_df = (pd.merge(top_df, top_df_counts_means)

          .sort_values(['rank_count', 'rank_mean'],

                       ascending=[False, True]))

top_df.iloc[:3, list(range(8))+ [-2, -1]]

num_queries = df['queryTime'].nunique()



summary = (df

           .groupby(['displayLink'], as_index=False)

           .agg({'rank': ['count', 'mean']})

           .sort_values(('rank', 'count'), ascending=False)

           .assign(coverage=lambda df: (df[('rank', 'count')].div(num_queries))))

summary.columns = ['displayLink', 'count', 'avg_rank', 'coverage']

summary['displayLink'] = summary['displayLink'].str.replace('www.', '')

summary['avg_rank'] = summary['avg_rank'].round(1)

summary['coverage'] = (summary['coverage'].mul(100)

                       .round(1).astype(str).add('%'))

summary.head(10)

print('number of queries:', num_queries)

fig = go.Figure()

fig.add_scatter(x=top_df['displayLink'].str.replace('www.', ''),

                y=top_df['rank'], mode='markers',

                marker={'size': 35, 'opacity': opacity},

                showlegend=False)

fig.layout.height = 600

fig.layout.yaxis.autorange = 'reversed'

fig.layout.yaxis.zeroline = False

iplot(fig)
rank_counts = (top_df

               .groupby(['displayLink', 'rank'])

               .agg({'rank': ['count']})

               .reset_index()

               .set_axis(['displayLink', 'rank', 'count'],

                         axis=1, inplace=False))

rank_counts[:15]

fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', ''),

                y=rank_counts['rank'], mode='text',

                marker={'color': '#000000'},

                text=rank_counts['count'], showlegend=False)

iplot(fig)
for domain in rank_counts['displayLink'].unique():

    rank_counts_subset = rank_counts[rank_counts['displayLink']==domain]

    fig.add_scatter(x=[domain.replace('www.', '')],

                    y=[11], mode='text',

                    marker={'size': 50},

                    text=str(rank_counts_subset['count'].sum()))

    fig.add_scatter(x=[domain.replace('www.', '')],

                    y=[12], mode='text',

                    text=format(rank_counts_subset['count'].sum() / num_queries, '.1%'))

    fig.add_scatter(x=[domain.replace('www.', '')],

                    y=[13], mode='text',

                    marker={'size': 50},

                    text=str(round(rank_counts_subset['rank']

                                   .mul(rank_counts_subset['count'])

                                   .sum() / rank_counts_subset['count']

                                   .sum(),2)))

fig.layout.title = ('Google Search Results Rankings<br>keyword(s): ' + 

                    ', '.join(list(df['searchTerms'].unique())) + 

                    ' | queries: ' + str(df['queryTime'].nunique()))

fig.layout.hovermode = False

fig.layout.yaxis.autorange = 'reversed'

fig.layout.yaxis.zeroline = False

fig.layout.yaxis.tickvals = list(range(1, 14))

fig.layout.yaxis.ticktext = list(range(1, 11)) + ['Total<br>appearances','Coverage', 'Avg. Pos.'] 

fig.layout.height = 700

fig.layout.width = 1200

fig.layout.yaxis.title = 'SERP Rank (number of appearances)'

fig.layout.showlegend = False

fig.layout.paper_bgcolor = '#eeeeee'

fig.layout.plot_bgcolor = '#eeeeee'

iplot(fig)
def plot_serps(df, opacity=0.1, num_domains=10, width=1200, height=700):

    """

    df: a DataFrame resulting from running advertools.serp_goog

    opacity: the opacity of the markers [0, 1]

    num_domains: how many domains to plot

    """

    top_domains = df['displayLink'].value_counts()[:num_domains].index.tolist()

    top_df = df[df['displayLink'].isin(top_domains)]

    top_df_counts_means = (top_df

                       .groupby('displayLink', as_index=False)

                       .agg({'rank': ['count', 'mean']})

                       .set_axis(['displayLink', 'rank_count', 'rank_mean'],

                                 axis=1, inplace=False))

    top_df = (pd.merge(top_df, top_df_counts_means)

          .sort_values(['rank_count', 'rank_mean'],

                       ascending=[False, True]))

    rank_counts = (top_df

               .groupby(['displayLink', 'rank'])

               .agg({'rank': ['count']})

               .reset_index()

               .set_axis(['displayLink', 'rank', 'count'],

                         axis=1, inplace=False))

    num_queries = df['queryTime'].nunique()

    fig = go.Figure()

    fig.add_scatter(x=top_df['displayLink'].str.replace('www.', ''),

                    y=top_df['rank'], mode='markers',

                    marker={'size': 35, 'opacity': opacity},

                    showlegend=False)

    fig.layout.height = 600

    fig.layout.yaxis.autorange = 'reversed'

    fig.layout.yaxis.zeroline = False

    fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', ''),

                y=rank_counts['rank'], mode='text',

                marker={'color': '#000000'},

                text=rank_counts['count'], showlegend=False)

    for domain in rank_counts['displayLink'].unique():

        rank_counts_subset = rank_counts[rank_counts['displayLink']==domain]

        fig.add_scatter(x=[domain.replace('www.', '')],

                        y=[11], mode='text',

                        marker={'size': 50},

                        text=str(rank_counts_subset['count'].sum()))

        fig.add_scatter(x=[domain.replace('www.', '')],

                        y=[12], mode='text',

                        text=format(rank_counts_subset['count'].sum() / num_queries, '.1%'))

        fig.add_scatter(x=[domain.replace('www.', '')],

                        y=[13], mode='text',

                        marker={'size': 50},

                        text=str(round(rank_counts_subset['rank']

                                       .mul(rank_counts_subset['count'])

                                       .sum() / rank_counts_subset['count']

                                       .sum(),2)))

    fig.layout.title = ('Google Search Results Rankings<br>keyword(s): ' + 

                        ', '.join(list(df['searchTerms'].unique())) + 

                        ' | queries: ' + str(df['queryTime'].nunique()))

    fig.layout.hovermode = False

    fig.layout.yaxis.autorange = 'reversed'

    fig.layout.yaxis.zeroline = False

    fig.layout.yaxis.tickvals = list(range(1, 14))

    fig.layout.yaxis.ticktext = list(range(1, 11)) + ['Total<br>appearances','Coverage', 'Avg. Pos.'] 

    fig.layout.height = height

    fig.layout.width = width

    fig.layout.yaxis.title = 'SERP Rank (number of appearances)'

    fig.layout.showlegend = False

    fig.layout.paper_bgcolor = '#eeeeee'

    fig.layout.plot_bgcolor = '#eeeeee'

    iplot(fig)
plot_serps(coffee_df, opacity=0.07, num_domains=15)
plot_serps(cafe_df, opacity=0.07, num_domains=8)