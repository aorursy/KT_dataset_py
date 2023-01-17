!pip install hdbscan
!pip install langdetect
import gensim
import hdbscan
import langdetect
import matplotlib.pyplot as plt
import numpy
import pandas
import plotly
import umap
import wordcloud
DATA_URL = 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv'

datadf = pandas.read_csv(DATA_URL)
print(datadf.info())
datadf.head()
%%time

def detect_language(text):
  try:
    return langdetect.detect(text)
  except:
    return None

datadf['language'] = datadf['title'].apply(detect_language)
print('Languages: %s' % datadf['language'].unique())
print(datadf[['language', 'cord_uid']].groupby('language')
                                      .count()
                                      .sort_values('cord_uid', ascending=False))
print('Licences: %s...' % datadf['license'].unique())
print(datadf[['license', 'cord_uid']].groupby('license')
                                     .count()
                                     .sort_values('cord_uid', ascending=False))
LANGUAGE = 'en'

docdf = datadf.loc[datadf['language'] == LANGUAGE, ['cord_uid', 'publish_time', 'title', 'authors', 'abstract']] \
              .rename(columns=dict(cord_uid='id', publish_time='date', abstract='text'))
docdf = docdf.groupby('id').first().reset_index()
docdf.sort_values('date', ascending=False, inplace=True)
docdf.dropna(inplace=True)

print(docdf.info())
docdf.head()
%%time

EPOCHS = 100

docs = [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text,
                                                                            deacc=True,
                                                                            min_len=3,
                                                                            max_len=100),
                                             [id])
        for id, text in docdf[['id', 'text']].values]

doc2vec = gensim.models.Doc2Vec(docs,
                                vector_size=300,
                                window=5,
                                min_count=5,
                                epochs=EPOCHS)
KEYWORD = 'covid'

vocab = list(doc2vec.wv.vocab)
vocab.sort()
print('%d words in vocabulary...' % len(vocab))
print('Sample: %s   ...   %s...\n' % (vocab[:100], vocab[-100:]))

print('\nWords similar with "%s":' % KEYWORD)
pandas.DataFrame(doc2vec.wv.most_similar(KEYWORD))
%%time

vecdf = pandas.DataFrame(doc2vec.wv.vectors, index=doc2vec.wv.index2word)
print(vecdf.info())
%%time

_umap = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0, metric='cosine')
umapdf = pandas.DataFrame(_umap.fit_transform(vecdf), index=vecdf.index)
print(umapdf.info())
%%time

_umap = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='cosine')
umap2ddf = pandas.DataFrame(_umap.fit_transform(vecdf), index=vecdf.index)
print(umap2ddf.info())
COLORS = doc2vec.wv.distances(KEYWORD)
ALPHA = 0.1

figure = plotly.subplots.make_subplots(rows=1, cols=2)
figure.add_trace(dict(type='histogram', x=COLORS, nbinsx=100),
                 1, 1)
figure.add_trace(dict(type='scatter', mode='markers', x=umap2ddf[0], y=umap2ddf[1], text=umap2ddf.index,
                      marker=dict(color=COLORS, colorscale='Plasma', showscale=True, opacity=ALPHA)),
                 1, 2)
figure.update_layout(width=1000, height=400, showlegend=False, margin=dict(l=0, t=0, r=0, b=0))
figure.show()
%%time

_hdbscan = hdbscan.HDBSCAN(min_cluster_size=15)
clusters = _hdbscan.fit_predict(umapdf)
unique_clusters = numpy.unique(clusters)

print('%d clusters...' % len(unique_clusters))
print('Clusters: %s...' % unique_clusters[:100])
data = [dict(type='histogram', x=clusters)]
layout=dict(width=1000, height=300, margin=dict(l=0, t=0, r=0, b=0))
figure = plotly.graph_objs.Figure(data=data, layout=layout)
figure.show()
ALPHA = 0.1

figure = plotly.subplots.make_subplots(rows=1, cols=2)
for cluster in unique_clusters:
  plotdf = umap2ddf[clusters == cluster]
  figure.add_trace(dict(type='scattergl', mode='markers', x=plotdf[0], y=plotdf[1],
                        marker=dict(opacity=ALPHA), name='cluster#%d' % cluster, text=plotdf.index),
                   1, 1)
  if cluster != -1:
    figure.add_trace(dict(type='scattergl', mode='markers', x=plotdf[0], y=plotdf[1],
                          marker=dict(opacity=ALPHA), name='cluster#%d' % cluster, text=plotdf.index),
                     1, 2)
figure.update_layout(width=1000, height=400, showlegend=False, margin=dict(l=0, t=0, r=0, b=0))
figure.show()
KEYWORD = 'covid'
NRESULTS = 10
NWORDS = 50

search_vector = doc2vec[KEYWORD]

rezdf = pandas.DataFrame(dict(context=[c for c in unique_clusters if c != -1]))
rezdf['score'] = doc2vec.wv.cosine_similarities(search_vector, [doc2vec.wv.vectors[clusters == c].mean(axis=0)
                                                                for c in unique_clusters if c != -1])
rezdf.sort_values('score', ascending=False, inplace=True)
rezdf = rezdf.head(NRESULTS)
rezdf['sample'] = rezdf['context'].apply(
    lambda context: ', '.join([w for w, score in doc2vec.wv.most_similar([doc2vec.wv.vectors[clusters == context].mean(axis=0)],
                                                                         topn=NWORDS)]))

for row in rezdf.to_dict(orient='record'):
    print('Score: %f, Topic ID: %d, sample: %s' % (row['score'], row['context'], row['sample']))
    cluster_mean_vector = doc2vec.wv.vectors[clusters == row['context']].mean(axis=0)
    similarities = dict(doc2vec.wv.most_similar([cluster_mean_vector], topn=NWORDS))
    plt.figure(figsize=(16, 4))
    plt.imshow(wordcloud.WordCloud(width=1600, height=400)
                        .generate_from_frequencies(similarities))
    plt.title('Cluster#%d' % row['context'], loc='left', fontsize=25, pad=20)
    plt.tight_layout()
    plt.show()
    print()

rezdf