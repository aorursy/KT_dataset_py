# the initial block is copied from creating_word_vectors_with_word2vec.ipynb

import nltk

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import *

import gensim

from gensim.models.word2vec import Word2Vec

from gensim.models.phrases import Phraser, Phrases

from sklearn.manifold import TSNE

import pandas as pd

from bokeh.io import output_notebook, output_file

from bokeh.plotting import show, figure

import string

%matplotlib inline
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('gutenberg')
from nltk.corpus import gutenberg
len(gutenberg.fileids())
gutenberg.fileids()
gberg_sent_tokens = sent_tokenize(gutenberg.raw())
gberg_sent_tokens[0:5]
gberg_sent_tokens[1]
word_tokenize(gberg_sent_tokens[1])
word_tokenize(gberg_sent_tokens[1])[14]
gberg_sents = gutenberg.sents()
gberg_sents[0:5]
len(gutenberg.words())
len(gutenberg.sents())
gberg_sents[5]
# CODE HERE

[w.lower() for w in gberg_sents[5]]
stpwrds = stopwords.words('english') + list(string.punctuation)
stpwrds
# CODE HERE

[w.lower() for w in gberg_sents[5] if w not in stpwrds]
stemmer = PorterStemmer()
# CODE HERE

[stemmer.stem(w.lower()) for w in gberg_sents[5] if w not in stpwrds]
phrases = Phrases(gberg_sents) # train detector
bigram = Phraser(phrases) # create a more efficient Phraser object for transforming sentences
bigram.phrasegrams # output count and score of each bigram
"Jon lives in New York City".split()
# CODE HERE

bigram["Jon lives in New York City".split()]
lower_sents = []

for s in gberg_sents:

    lower_sents.append( [w.lower() for w in s if w not in list(string.punctuation)] )
lower_sents[0:5]
lower_bigram = Phraser(Phrases(lower_sents))
lower_bigram.phrasegrams # miss taylor, mr woodhouse, mr weston
lower_bigram["jon lives in new york city".split()]
lower_bigram = Phraser(Phrases(lower_sents, min_count=32, threshold=64))

lower_bigram.phrasegrams
# as in Maas et al. (2001):

# - leave in stop words ("indicative of sentiment")

# - no stemming ("model learns similar representations of words of the same stem when data suggests it")

clean_sents = []

for s in lower_sents:

    clean_sents.append(lower_bigram[s])
clean_sents[0:9]
clean_sents[6] # could consider removing stop words or common words
# max_vocab_size can be used instead of min_count (which has increased here)

model = Word2Vec(sentences=clean_sents, size=64, sg=1, window=10, min_count=10, seed=42, workers=8)

model.save('../clean_gutenberg_model.w2v')
# skip re-training the model with the next line:  

model = gensim.models.Word2Vec.load('../clean_gutenberg_model.w2v')
len(model.wv.vocab) # 17k with raw data
len(model['dog'])
model['dog']
model.most_similar('dog')
model.most_similar('think')
model.most_similar('day')
model.doesnt_match("morning afternoon evening dog".split())
model.similarity('morning', 'dog')
model.most_similar('ma_am') 
model.most_similar(positive=['father', 'woman'], negative=['man']) 
tsne = TSNE(n_components=2, n_iter=1000)
X_2d = tsne.fit_transform(model[model.wv.vocab])
coords_df = pd.DataFrame(X_2d, columns=['x','y'])

coords_df['token'] = model.wv.vocab.keys()
coords_df.to_csv('../clean_gutenberg_tsne.csv', index=False)
coords_df = pd.read_csv('../clean_gutenberg_tsne.csv')
coords_df.head()
_ = coords_df.plot.scatter('x', 'y', figsize=(12,12), marker='.', s=10, alpha=0.2)
output_notebook()
subset_df = coords_df.sample(n=5000)
p = figure(plot_width=800, plot_height=800)

_ = p.text(x=subset_df.x, y=subset_df.y, text=subset_df.token)
show(p)
# output_file() here