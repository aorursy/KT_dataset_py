import nltk

from nltk import word_tokenize, sent_tokenize

import gensim

from gensim.models.word2vec import Word2Vec

from sklearn.manifold import TSNE

import pandas as pd

from bokeh.io import output_notebook, output_file

from bokeh.plotting import show, figure

%matplotlib inline
nltk.download('punkt')
import string

from nltk.corpus import stopwords

from nltk.stem.porter import *

from gensim.models.phrases import Phraser, Phrases

from keras.preprocessing.text import one_hot
nltk.download('stopwords')
nltk.download('gutenberg')
from nltk.corpus import gutenberg

gberg_sents = gutenberg.sents()
gberg_sents[4]
[w.lower() for w in gberg_sents[4]]
stpwrds = stopwords.words('english') + list(string.punctuation)
stpwrds
[w.lower() for w in gberg_sents[4] if w not in stpwrds]
stemmer = PorterStemmer()
[stemmer.stem(w.lower()) for w in gberg_sents[4] if w not in stpwrds]
phrases = Phrases(gberg_sents)
bigram = Phraser(phrases)
bigram.phrasegrams 
"Jon lives in New York City".split()
bigram["Jon lives in New York City".split()]
lower_sents = []

for s in gberg_sents:

    lower_sents.append([w.lower() for w in s if w not in list(string.punctuation)])
lower_sents[0:5]
lower_bigram = Phraser(Phrases(lower_sents))
lower_bigram.phrasegrams
lower_bigram["jon lives in new york city".split()]
lower_bigram = Phraser(Phrases(lower_sents, min_count=32, threshold=64))

lower_bigram.phrasegrams
clean_sents = []

for s in lower_sents:

    clean_sents.append(lower_bigram[s])
clean_sents[0:9]
clean_sents[6]
model = Word2Vec(sentences=clean_sents, size=64, sg=1, window=10, min_count=10, seed=42, workers=8)

model.save('clean_gutenberg_model.w2v')
model = gensim.models.Word2Vec.load('../input/dataset/clean_gutenberg_model.w2v')
len(model.wv.vocab)
model['ma_am']
model.most_similar('ma_am') 
model.most_similar(positive=['ma_am', 'man'], negative=['woman'])
model.most_similar(positive=['father', 'woman'], negative=['man']) 
tsne = TSNE(n_components=2, n_iter=1000)
X_2d = tsne.fit_transform(model[model.wv.vocab])
coords_df = pd.DataFrame(X_2d, columns=['x','y'])

coords_df['token'] = model.wv.vocab.keys()
coords_df.head()
coords_df.to_csv('clean_gutenberg_tsne.csv', index=False)
coords_df = pd.read_csv('../input/dataset/clean_gutenberg_tsne.csv')
_ = coords_df.plot.scatter('x', 'y', figsize=(12,12), marker='.', s=10, alpha=0.2)