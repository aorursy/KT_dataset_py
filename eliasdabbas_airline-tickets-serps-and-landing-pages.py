!pip install advertools
import advertools as adv

import pandas as pd

pd.options.display.max_columns = None

import plotly.graph_objects as go

import plotly

print('advertools v' + str(adv.__version__))

print('pandas v' + str(pd.__version__))

print('plotly v' + str(plotly.__version__))

flights = pd.read_csv('/kaggle/input/flights-serps-and-landing-pages/flights_serp_scrape.csv')

flights.head(2)
serps_to_plot = flights.copy()

serps_to_plot.columns = flights.columns.str.replace('serp_', '').str.replace('scrape_', '')

def plot_data(serps_to_plot, num_domains=10, select_domain=None):

#     df = pd.DataFrame(serp_results, columns=serp_results[0].keys())

    df = serps_to_plot

    if select_domain:

        df = df[df['displayLink'].isin(select_domain)]

    top_domains = df['displayLink'].value_counts()[:num_domains].index.tolist()

    top_df = df[df['displayLink'].isin(top_domains)]

    top_df_counts_means = (top_df

                           .groupby('displayLink', as_index=False)

                           .agg({'rank': ['count', 'mean']}))

    top_df_counts_means.columns = ['displayLink', 'rank_count', 'rank_mean']

    top_df = (pd.merge(top_df, top_df_counts_means)

              .sort_values(['rank_count', 'rank_mean'],

                           ascending=[False, True]))

    rank_counts = (top_df

                   .groupby(['displayLink', 'rank'])

                   .agg({'rank': ['count']})

                   .reset_index())

    rank_counts.columns = ['displayLink', 'rank', 'count']

    summary = (df

               .groupby(['displayLink'], as_index=False)

               .agg({'rank': ['count', 'mean']})

               .sort_values(('rank', 'count'), ascending=False)

               .assign(coverage=lambda df: (df[('rank', 'count')]

                                            .div(df[('rank', 'count')]

                                                 .sum()))))

    summary.columns = ['displayLink', 'count', 'avg_rank', 'coverage']

    summary['displayLink'] = summary['displayLink'].str.replace('www.', '')

    summary['avg_rank'] = summary['avg_rank'].round(1)

    summary['coverage'] = (summary['coverage'].mul(100)

                           .round(1).astype(str).add('%'))

    num_queries = df['queryTime'].nunique()



    fig = go.Figure()

    fig.add_scatter(x=top_df['displayLink'].str.replace('www.', ''),

                    y=top_df['rank'], mode='markers',

                    marker={'size': 30, 'opacity': 1/rank_counts['count'].max()})



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

                        text=format(rank_counts_subset['count'].sum() / num_queries, '.1%'))

        fig.add_scatter(x=[domain.replace('www.', '')],

                        y=[-2], mode='text',

                        marker={'size': 50},

                        text=str(round(rank_counts_subset['rank']

                                       .mul(rank_counts_subset['count'])

                                       .sum() / rank_counts_subset['count']

                                       .sum(), 2)))



    minrank, maxrank = min(top_df['rank'].unique()), max(top_df['rank'].unique())

    fig.layout.yaxis.tickvals = [-2, -1, 0] + list(range(minrank, maxrank+1))

    fig.layout.yaxis.ticktext = ['Avg. Pos.', 'Coverage', 'Total<br>appearances'] + list(range(minrank, maxrank+1))



#     fig.layout.height = max([600, 100 + ((maxrank - minrank) * 50)])

    fig.layout.height = 600

    fig.layout.yaxis.title = 'SERP Rank (number of appearances)'

    fig.layout.showlegend = False

    fig.layout.paper_bgcolor = '#eeeeee'

    fig.layout.plot_bgcolor = '#eeeeee'

    fig.layout.autosize = True

    fig.layout.margin.r = 2

    fig.layout.margin.l = 120

    fig.layout.margin.pad = 0

    fig.layout.hovermode = False

    fig.layout.yaxis.autorange = 'reversed'

    fig.layout.yaxis.zeroline = False

    return fig



fig = plot_data(serps_to_plot.query('gl == "us"'))

fig.layout.title = 'SERP Postions for USA'

fig
fig = plot_data(serps_to_plot.query('gl == "uk"'))

fig.layout.title = 'SERP Postions for UK'

fig
(flights

 .drop_duplicates(subset=['serp_searchTerms', 'serp_gl'])

 [['serp_searchTerms', 'serp_gl', 'serp_totalResults']]

 .sort_values('serp_totalResults', ascending=False)[:20]

 .reset_index(drop=True)

 .style.format({'serp_totalResults': '{:,}'})

 .set_caption('Queries by number of results'))
flights['scrape_body_text'].str.contains('[kc]orona', regex=True, case=False).mean()
top_h1_tags = flights['scrape_h1'].value_counts()

top_h1_tags[2:12]
(flights[flights['scrape_h1']=='Flights to Vienna']

 [['serp_searchTerms', 'serp_link', 'scrape_h1']]

 .sort_values('serp_link')

 .style.set_caption('Pages with "Flights to Vienna" as their H1 tag'))
flights[flights['serp_searchTerms'].str.contains('vienna')][['serp_searchTerms', 'serp_link', 'scrape_h1']].sort_values('serp_link')
(flights['scrape_h1'].dropna()

 .str.split('@@').str.len()

 .value_counts(normalize=False)

 .to_frame().reset_index()

 .rename(columns={'index': 'h1_tags_per_page',

                  'scrape_h1': 'count'})

 .assign(perc=lambda df: df['count'].div(df['count'].sum()))

 .style.format({'perc': '{:.1%}'})

.hide_index())
flights[flights['scrape_h1'].str.split('@@').str.len() > 2].sort_values(['serp_displayLink', 'serp_link'])['scrape_h1'].str.split('@@')[:15]
[f for f in dir(adv) if f.startswith('extract')]
serp_title_currency = adv.extract_currency(flights['serp_title'])

serp_title_currency.keys()
serp_title_currency['overview']
serp_title_currency['top_currency_symbols']
{sym[0] for sym in serp_title_currency['currency_symbol_names'] if sym}
[x for x in serp_title_currency['surrounding_text'] if x][:15]
snippet_emoji = adv.extract_emoji(flights['serp_snippet'])

snippet_emoji.keys()
snippet_emoji['overview']
snippet_emoji['top_emoji']
snippet_questions = adv.extract_questions(flights['serp_snippet'])

snippet_questions.keys()
snippet_questions['overview']
pd.Series(' '.join(q) for q in snippet_questions['question_text'] if q).value_counts()[:10]
adv.extract_hashtags(flights['scrape_body_text'].dropna())['overview']
body_text_length = flights['scrape_body_text'].dropna().str.split().str.len()
fig = go.Figure()

fig.add_histogram(x=body_text_length.values)

fig.layout.title = 'Distribution of word count of body text'

fig.layout.bargap = 0.1

fig.layout.xaxis.title = 'Number of words per page'

fig.layout.yaxis.title = 'Count'

fig
fig = go.Figure()

fig.add_histogram(x=body_text_length[body_text_length < 1700].values)

fig.layout.title = 'Distribution of word count of body text (for pages having less than 1,700 words)'

fig.layout.bargap = 0.1

fig.layout.xaxis.title = 'Number of words per page'

fig.layout.yaxis.title = 'Count'

fig