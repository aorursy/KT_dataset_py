! pip install nlplot
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nlplot

from plotly.subplots import make_subplots

import plotly.express as px



pd.set_option('display.max_columns', 300)

pd.set_option('display.max_rows', 300)

pd.options.display.float_format = '{:.3f}'.format

pd.set_option('display.max_colwidth', 5000)
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
# In this case, we're going to sample and visualize 10000 pieces of data

train = train.sample(n=10000, random_state=0)
# Convert text to lowercase

train['text'] = train['text'].apply(lambda x: x.lower())
display(train.head(), train.shape)
df = train.groupby('sentiment').size().reset_index(name='count')

fig = px.bar(df, y='count', x='sentiment', text='count')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(

    title=str('sentiment counts'),

    xaxis_title=str('sentiment'),

    width=700,

    height=500,

    )

fig.show()
# initialize

npt = nlplot.NLPlot(train, target_col='text')

npt_negative = nlplot.NLPlot(train.query('sentiment == "negative"'), target_col='text')

npt_neutral = nlplot.NLPlot(train.query('sentiment == "neutral"'), target_col='text')

npt_positive = nlplot.NLPlot(train.query('sentiment == "positive"'), target_col='text')
stopwords = npt.get_stopword(top_n=30, min_freq=0)

print(stopwords)
# uni-gram

npt.bar_ngram(

    title='uni-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=1,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)
# bi-gram

npt.bar_ngram(

    title='bi-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=2,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)
# tri-gram

npt.bar_ngram(

    title='tri-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=3,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)
# positive/neutral/negative

fig_unigram_positive = npt_positive.bar_ngram(

    title='uni-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=1,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)



fig_unigram_neutral = npt_neutral.bar_ngram(

    title='uni-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=1,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)



fig_unigram_negative = npt_negative.bar_ngram(

    title='uni-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=1,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)
# subplot

trace1 = fig_unigram_positive['data'][0]

trace2 = fig_unigram_neutral['data'][0]

trace3 = fig_unigram_negative['data'][0]



fig = make_subplots(rows=1, cols=3, subplot_titles=('positive', 'neutral', 'negative'), shared_xaxes=False)

fig.update_xaxes(title_text='word count', row=1, col=1)

fig.update_xaxes(title_text='word count', row=1, col=2)

fig.update_xaxes(title_text='word count', row=1, col=3)



fig.update_layout(height=1100, width=1000, title_text='unigram positive vs neutral vs negative')

fig.add_trace(trace1, row=1, col=1)

fig.add_trace(trace2, row=1, col=2)

fig.add_trace(trace3, row=1, col=3)



fig.show()
# positive/neutral/negative

fig_bigram_positive = npt_positive.bar_ngram(

    title='bi-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=2,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)



fig_bigram_neutral = npt_neutral.bar_ngram(

    title='bi-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=2,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)



fig_bigram_negative = npt_negative.bar_ngram(

    title='bi-gram',

    xaxis_label='word_count',

    yaxis_label='word',

    ngram=2,

    top_n=50,

    width=800,

    height=1100,

    stopwords=stopwords,

)
# subplot

trace1 = fig_bigram_positive['data'][0]

trace2 = fig_bigram_neutral['data'][0]

trace3 = fig_bigram_negative['data'][0]



fig = make_subplots(rows=1, cols=3, subplot_titles=('positive', 'neutral', 'negative'), shared_xaxes=False)

fig.update_xaxes(title_text='word count', row=1, col=1)

fig.update_xaxes(title_text='word count', row=1, col=2)

fig.update_xaxes(title_text='word count', row=1, col=3)



fig.update_layout(height=1100, width=1000, title_text='bigram positive vs neutral vs negative')

fig.add_trace(trace1, row=1, col=1)

fig.add_trace(trace2, row=1, col=2)

fig.add_trace(trace3, row=1, col=3)



fig.show()
npt.treemap(

    title='All sentiment Tree of Most Common Words',

    ngram=1,

    stopwords=stopwords,

)
npt_positive.treemap(

    title='Positive Tree of Most Common Words',

    ngram=1,

    stopwords=stopwords,

)
npt_neutral.treemap(

    title='Neutral Tree of Most Common Words',

    ngram=1,

    stopwords=stopwords,

)
npt_negative.treemap(

    title='Negative Tree of Most Common Words',

    ngram=1,

    stopwords=stopwords,

)
npt.word_distribution(

    title='number of words distribution'

)
fig_wd_positive = npt_positive.word_distribution()

fig_wd_neutral = npt_neutral.word_distribution()

fig_wd_negative = npt_negative.word_distribution()
trace1 = fig_wd_positive['data'][0]

trace2 = fig_wd_neutral['data'][0]

trace3 = fig_wd_negative['data'][0]



fig = make_subplots(rows=3, cols=1, subplot_titles=('positive', 'neutral', 'negative'), shared_xaxes=True)



fig.update_layout(height=1200, width=900, title_text='words distribution positive vs neutral vs negative')

fig.add_trace(trace1, row=1, col=1)

fig.add_trace(trace2, row=2, col=1)

fig.add_trace(trace3, row=3, col=1)



fig.show()
# All sentiment

npt.wordcloud(

    stopwords=stopwords,

    colormap='tab20_r',

)
# positive

npt_positive.wordcloud(

    stopwords=stopwords,

    colormap='tab20_r',

)
# neutral

npt_neutral.wordcloud(

    stopwords=stopwords,

    colormap='tab20_r',

)
# negative

npt_negative.wordcloud(

    stopwords=stopwords,

    colormap='tab20_r',

)
npt.build_graph(stopwords=stopwords, min_edge_frequency=25)

npt_positive.build_graph(stopwords=stopwords, min_edge_frequency=10)

npt_neutral.build_graph(stopwords=stopwords, min_edge_frequency=10)

npt_negative.build_graph(stopwords=stopwords, min_edge_frequency=10)
# graph data

display(

    npt.node_df.head(),

    npt.edge_df.head(),

)
# all data

npt.co_network(

    title='All sentiment Co-occurrence network',

    color_palette='hls',

    width=1000,

    height=1200,

)
npt_positive.co_network(

    title='Positive Co-occurrence network',

    color_palette='hls',

    width=1000,

    height=1200,

)
npt_neutral.co_network(

    title='Neutral Co-occurrence network',

    color_palette='hls',

    width=1000,

    height=1200,

)
npt_negative.co_network(

    title='Negative Co-occurrence network',

    color_palette='hls',

    width=1000,

    height=1200,

)
npt.sunburst(

    title='All sentiment sunburst chart',

    colorscale=True,

    color_continuous_scale='Oryel',

    width=1000,

    height=800,

)
npt_positive.sunburst(

    title='Positive sunburst chart',

    colorscale=True,

    color_continuous_scale='Oryel',

    width=1000,

    height=800,

)
npt_neutral.sunburst(

    title='Neutral sunburst chart',

    colorscale=True,

    color_continuous_scale='Oryel',

    width=1000,

    height=800,

)
npt_negative.sunburst(

    title='Negative sunburst chart',

    colorscale=True,

    color_continuous_scale='Oryel',

    width=1000,

    height=800,

)
npt.ldavis(num_topics=3, passes=5, save=False)