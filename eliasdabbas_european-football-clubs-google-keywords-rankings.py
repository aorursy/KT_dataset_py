import advertools as adv

import pandas as pd

pd.options.display.max_columns = None

from plotly.tools import make_subplots

import plotly.graph_objs as go

import plotly

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



print('Package         Version')

print('=' * 25)

for package in [plotly, pd, adv]:

    print(f'{package.__name__:<15}', ': ', package.__version__, sep='')
# page = 'https://en.wikipedia.org/wiki/List_of_UEFA_club_competition_winners'

# column_key = pd.read_html(page)[0]

# column_key = column_key.rename(columns={0: 'abbreviation', 1: 'tournament'})

# column_key.to_csv('column_key.csv', index=False)

# clubs = pd.read_html(page)[1]

# clubs.to_csv('clubs.csv', index=False)
column_key = pd.read_csv('../input/column_key.csv')

column_key
clubs = pd.read_csv('../input/clubs.csv')

clubs.head(10)
top_countries = (clubs

                 .groupby('Country')

                 .agg({'Total': 'sum'})

                 .sort_values('Total', ascending=False)

                 .reset_index()

                 .head(10))

top_countries
(clubs

 .groupby(['Country'])

 .agg({'Club': 'count', 'Total': 'sum'})

 .sort_values('Club', ascending=False)

 .reset_index()

 .head(9)

 .set_axis(['country', 'num_clubs', 'total_wins'], axis=1, inplace=False)

 .assign(wins_per_club=lambda df: df['total_wins'].div(df['num_clubs']))

 .style.background_gradient(high=0.2))



clubs_list = clubs['Club'].str.lower().tolist()

clubs_list[:10]
lang_football = {'en': 'football',

                 'fr': 'football',

                 'de': 'fußball',

                 'es': 'fútbol',

                 'it': 'calcio',

                 'pt-BR': 'futebol',

                 'nl': 'voetbal'}

lang_football
# cx = 'YOUR_CX_FROM_GOOGLE'

# key = 'YOUR_GOOGLE_DEV_KEY'



# serp_dfs = []

# for lang, q in lang_football.items():

#     temp_serp = adv.serp_goog(cx=cx, key=key, 

#                               hl=lang,

#                               q=[club + ' ' + q for club in clubs_list])

#     serp_dfs.append(temp_serp)



# serp_clubs = pd.concat(serp_dfs, sort=False)

# serp_clubs.to_csv('serp_clubs.csv', index=False)
serp_clubs = pd.read_csv('../input/serp_clubs.csv', parse_dates=['queryTime'])

print(serp_clubs.shape)

serp_clubs.head()
club_country = {club.lower(): country.lower() for club, country in zip(clubs['Club'], clubs['Country'])}

football_multi = '|'.join([' ' + football for football in lang_football.values()])



serp_clubs['country'] = [club_country[club].title()

                         for club in serp_clubs['searchTerms'].str.replace(football_multi, '')]

serp_clubs['club'] = serp_clubs['searchTerms'].str.replace(football_multi, '').str.title()

serp_clubs[['searchTerms', 'country', 'club']].sample(10)
print('unique domains:', serp_clubs['displayLink'].nunique())

print('number of results:', serp_clubs.__len__())

serp_clubs['displayLink'].value_counts().reset_index()[:10]
serp_clubs[serp_clubs['club']=='Barcelona']['displayLink'].value_counts().reset_index()[:10]
serp_clubs[serp_clubs['hl']=='de']['displayLink'].value_counts().reset_index()[:10]
serp_clubs[serp_clubs['country']=='Italy']['displayLink'].value_counts().reset_index()[:10]
serp_clubs['link'].value_counts().reset_index()[:10]
adv.extract_urls(serp_clubs['link'])['top_tlds'][:10]
(pd.DataFrame({

    'tld': [x[0] for x in  adv.extract_urls(serp_clubs['link'])['top_tlds']],

    'freq': [x[1] for x in  adv.extract_urls(serp_clubs['link'])['top_tlds']]

}).assign(percentage=lambda df: df['freq'].div(df['freq'].sum()),

          cumsum=lambda df: df['freq'].cumsum(), 

          cum_perc=lambda df: df['cumsum'].div(df['freq'].sum()))

 .head(15)

 .style.format({'percentage': '{:.2%}', 'cumsum': '{:,}', 'cum_perc': '{:.2%}'}))
adv.word_frequency(serp_clubs['title'],

                   rm_words=adv.stopwords['english'].union(['-', '|', '  ', ''])).head(10)
adv.word_frequency(serp_clubs[serp_clubs['hl']=='nl']['title'],

                   rm_words=adv.stopwords['english'].union(['-', '|', '  ', ''])).head(10)
serp_clubs['snippet'].isna().sum(), serp_clubs['title'].isna().sum()
serp_clubs[serp_clubs['snippet'].isna()]['displayLink'].value_counts()
adv.word_frequency(serp_clubs['snippet'].fillna(''),

                   rm_words=adv.stopwords['english'].union(['-', '|', '  ','·', '', 'de'])).head(15)
adv.word_frequency(serp_clubs[serp_clubs['hl']=='en']['snippet'].fillna(''),

                   rm_words=adv.stopwords['english'].union(['-', '|', '  ','·', '', 'de'])).head(15)
adv.word_frequency(serp_clubs[serp_clubs['club']=='Liverpool']['snippet'].fillna(''),

                   phrase_len=2,

                   rm_words=adv.stopwords['english'].union([ '|', '', 'de'])).head(20)
(serp_clubs

 .drop_duplicates(['searchTerms'])

 .groupby('searchTerms', as_index=False)

 .agg({'totalResults': 'sum'})

 .sort_values('totalResults', ascending=False)

 .reset_index(drop=True)

 [:10]

 .style.format({'totalResults': '{:,}'}))
(serp_clubs

 .drop_duplicates(['searchTerms'])

 .groupby('club', as_index=False)

 .agg({'totalResults': 'sum'})

 .sort_values('totalResults', ascending=False)

 .reset_index(drop=True)

 [:10]

 .style.format({'totalResults': '{:,}'}))
fig = make_subplots(1, 7, print_grid=False, shared_yaxes=True)

for i, lang in enumerate(serp_clubs['hl'].unique()[:7]):

    df = serp_clubs[serp_clubs['hl']==lang]

    

    fig.append_trace(go.Bar(y=df['displayLink'].value_counts().values[:8], 

                            x=df['displayLink'].value_counts().index.str.replace('www.', '')[:8],

                            name=lang,

                            orientation='v'), row=1, col=i+1)





fig.layout.margin = {'b': 150, 'r': 30}

fig.layout.legend.orientation = 'h'

fig.layout.legend.y = -0.5

fig.layout.legend.x = 0.15

fig.layout.title = 'Top Domains by Language of Search'

fig.layout.yaxis.title = 'Number of Appearances on SERPs'

fig.layout.plot_bgcolor = '#eeeeee'

fig.layout.paper_bgcolor = '#eeeeee'

iplot(fig)
fig = make_subplots(1, 7, shared_yaxes=True, print_grid=False)

for i, country in enumerate(serp_clubs['country'].unique()[:7]):

    if country in top_countries['Country'][:7].values:

        df = serp_clubs[serp_clubs['country']==country]



        fig.append_trace(go.Bar(y=df['displayLink'].value_counts().values[:8], 

                                x=df['displayLink'].value_counts().index.str.replace('www.', '')[:8],

                                name=country,

                                orientation='v'), row=1, col=i+1)



fig.layout.margin = {'b': 150, 'r': 0}

fig.layout.legend.orientation = 'h'

fig.layout.legend.y = -0.5

fig.layout.legend.x = 0.15

fig.layout.title = 'Top Domains by Country of Club'

fig.layout.yaxis.title = 'Number of Appearances on SERPs'

fig.layout.plot_bgcolor = '#eeeeee'

fig.layout.paper_bgcolor = '#eeeeee'

iplot(fig)
def plot_serps(df, opacity=0.1, num_domains=10, width=None, height=700):

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

#     fig.layout.title = ('Google Search Results Rankings<br>keyword(s): ' + 

#                         ', '.join(list(df['searchTerms'].unique()[:5])) + 

#                         str(df['queryTime'].nunique()) + ' Football (Soccer) Queries')

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

    return fig
fig = plot_serps(serp_clubs, opacity=0.05)

fig.layout.title = 'SERPs for "<club_name> football" (79 clubs)'

iplot(fig)
fig = plot_serps(serp_clubs[serp_clubs['hl']=='es'], opacity=0.15)

fig.layout.title = 'SERPs for "<club_name> fútbol" in Spanish (79 clubs)'

iplot(fig)
fig = plot_serps(serp_clubs[serp_clubs['hl']=='en'], opacity=0.15)

fig.layout.title = 'SERPs for "<club_name> football" in English (79 clubs)'

iplot(fig)
fig = plot_serps(serp_clubs[serp_clubs['hl']=='de'], opacity=0.15)

fig.layout.title = 'SERPs for "<club_name> fußball" in German (79 clubs)'

iplot(fig)
fig = plot_serps(serp_clubs[serp_clubs['club']=='Liverpool'], opacity=0.15, num_domains=15)

fig.layout.title = 'SERPs for "liverpool football"'

iplot(fig)