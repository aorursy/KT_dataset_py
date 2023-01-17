!pip install nagisa

!pip install nlplot
import glob

import os



import nagisa

import nlplot

import pandas as pd
def extract_txt(filename: str) -> str:

    with open(filename) as text_file:

        # 0: URL, 1: timestamp

        text = text_file.readlines()[2:]

        text = [sentence.strip() for sentence in text]

        text = list(filter(lambda line: line != '', text))

        return ''.join(text)





EXTRACTDIR = '/kaggle/input/livedoor-news/'

categories = [

    name for name

    in os.listdir(os.path.join(EXTRACTDIR, "text"))

    if os.path.isdir(os.path.join(EXTRACTDIR, "text", name))]



categories = sorted(categories)

table = str.maketrans({

    '\n': '',

    '\t': '　',

    '\r': '',

})



all_text = []

all_label = []



for cat in categories:

    files = glob.glob(os.path.join(EXTRACTDIR, "text", cat, "{}*.txt".format(cat)))

    files = sorted(files)

    body = [extract_txt(elem).translate(table) for elem in files]

    label = [cat] * len(body)



    all_text.extend(body)

    all_label.extend(label)



df = pd.DataFrame({'text': all_text, 'label': all_label})
df.head()
df = df.loc[:10]

tagger = nagisa.Tagger()

df['sep_text'] = [tagger.extract(text, extract_postags=['名詞']).words for text in df['text']]
npt = nlplot.NLPlot(df, target_col='sep_text')

stopwords = npt.get_stopword(top_n=5, min_freq=0)
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
npt.build_graph(stopwords=stopwords, min_edge_frequency=5)

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