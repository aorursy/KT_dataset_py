import nltk
from nltk import word_tokenize, sent_tokenize
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
%matplotlib inline
from nltk.corpus import gutenberg
len(gutenberg.fileids())
gutenberg.fileids()
# a convenient method that handles newlines, as well as tokenizing sentences and words in one shot
gberg_sents = gutenberg.sents()
gberg_sents[4]
model = Word2Vec(sentences=gberg_sents, size=64, sg=1, window=10, min_count=5, seed=42)
model.save('raw_gutenberg_model.w2v')
# skip re-training the model with the next line:  
model = gensim.models.Word2Vec.load('raw_gutenberg_model.w2v')
model['dog']
len(model['dog'])
model.most_similar('dog') # distance
model.similarity('father', 'dog')
model.wv.vocab
len(model.wv.vocab)
X = model[model.wv.vocab]
tsne = TSNE(n_components=2, n_iter=1000) # 200 is minimum iter; default is 1000
X_2d = tsne.fit_transform(X)
X_2d[0:5]
# create DataFrame for storing results and plotting
coords_df = pd.DataFrame(X_2d, columns=['x','y'])
coords_df['token'] = model.wv.vocab.keys()
coords_df.head()
coords_df.to_csv('raw_gutenberg_tsne.csv', index=False)
#code commented to save the time
#coords_df = pd.read_csv('raw_gutenberg_tsne.csv')
#_ = coords_df.plot.scatter('x', 'y', figsize=(12,12), marker='.', s=10, alpha=0.2)
#output_notebook() # output bokeh plots inline in notebook
#subset_df = coords_df.sample(n=5000)
#p = figure(plot_width=800, plot_height=800)
#_ = p.text(x=subset_df.x, y=subset_df.y, text=subset_df.token)
#show(p)