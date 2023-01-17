!pip install nagisa

!pip install git+https://github.com/takapy0210/nlplot.git
import numpy as np

import pandas as pd

import nlplot

import nagisa
df = pd.read_csv('../input/aibo.csv')
df.head()
df.shape
df.groupby('seasons').count()['vols'].plot.bar()
add_noun = [

    '水谷豊',

    '寺脇康文',

    '及川光博',

    '成宮寛貴',

    '反町隆史',

    '川原和久',

    '六角精児',

    '鈴木砂羽',

    '岸部一徳',

    '原田龍二',

    '真飛聖',

    '鈴木杏樹',

    '山西惇',

    '神保悟志'

]



tagger = nagisa.Tagger(single_word_list=add_noun)
df['sep_descs'] = [tagger.extract(text, extract_postags=['名詞']).words for text in df['descs']]
df.head()
npt = nlplot.NLPlot(df, taget_col='sep_descs')
# stopwords = npt.get_stopword(top_n=30, min_freq=0)

stopwords = [str(i) for i in range(30)]

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
npt.treemap(

    title='Tree Map',

    ngram=1,

    stopwords=stopwords,

)
npt.word_distribution(

    title='number of words distribution'

)
npt.wordcloud(

    stopwords=stopwords,

    colormap='tab20_r',

)
npt.build_graph(stopwords=stopwords, min_edge_frequency=25)

npt.co_network(

    title='Co-occurrence network',

    color_palette='hls',

    width=1000,

    height=1200,

)
npt.sunburst(

    title='sunburst chart',

    colorscale=True,

    color_continuous_scale='Oryel',

    width=1000,

    height=800,

)
npt.ldavis(num_topics=3, passes=5, save=False)