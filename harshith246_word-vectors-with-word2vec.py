import nltk

from nltk import word_tokenize, sent_tokenize

import gensim

from gensim.models.word2vec import Word2Vec

from sklearn.manifold import TSNE

import pandas as pd

from bokeh.io import output_notebook

from bokeh.plotting import show, figure

%matplotlib inline
nltk.download('punkt') # English-language sentence tokenizer (not all periods end sentences; not all sentences start with a capital letter)
nltk.download('gutenberg')
from nltk.corpus import gutenberg
len(gutenberg.fileids())
gutenberg.fileids()
gberg_sent_tokens = sent_tokenize(gutenberg.raw())
gberg_sent_tokens[0:5]
gberg_sent_tokens[1]
word_tokenize(gberg_sent_tokens[1])
word_tokenize(gberg_sent_tokens[1])[14]
# a convenient method that handles newlines, as well as tokenizing sentences and words in one shot

gberg_sents = gutenberg.sents()
gberg_sents[0:5]
gberg_sents[4]
gberg_sents[4][14]
# another convenient method that we don't immediately need: 

gutenberg.words() 
# gutenberg.words() is analogous to the following line, which need not be run: 

# word_tokenize(gutenberg.raw())
# our Gutenberg corpus is 2.6m words in length: 

len(gutenberg.words())
model = Word2Vec(sentences=gberg_sents, size=64, sg=1, window=10, min_count=5, seed=42, workers=8)
model.save('raw_gutenberg_model.w2v')
# skip re-training the model with the next line:  

model = gensim.models.Word2Vec.load('raw_gutenberg_model.w2v')
model['dog']
len(model['dog'])
model.most_similar('dog') # distance
model.most_similar('think')
model.most_similar('day')
model.most_similar('father')
model.doesnt_match("mother father daughter dog".split())
model.similarity('father', 'dog')
# close, but not quite; distinctly in female direction: 

model.most_similar(positive=['father', 'woman'], negative=['man']) 
# more confident about this one: 

model.most_similar(positive=['son', 'woman'], negative=['man']) 
model.most_similar(positive=['husband', 'woman'], negative=['man']) 
model.most_similar(positive=['king', 'woman'], negative=['man'], topn=30) 
# impressive for such a small data set, without any cleaning, e.g., to lower case (covered next)
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
coords_df = pd.read_csv('raw_gutenberg_tsne.csv')
_ = coords_df.plot.scatter('x', 'y', figsize=(12,12), marker='.', s=10, alpha=0.2)
output_notebook() # output bokeh plots inline in notebook
subset_df = coords_df.sample(n=5000)
p = figure(plot_width=800, plot_height=800)

_ = p.text(x=subset_df.x, y=subset_df.y, text=subset_df.token)
show(p)