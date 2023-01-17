import pandas as pd
import csv
from nltk.tokenize.casual import TweetTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
df = pd.read_csv('../input/chat.csv',  delimiter=' ',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
corpus = []
deu_stops = stopwords.words('german')
sno = nltk.stem.SnowballStemmer('german')
for index, row in df.iterrows():
    text = row['text'].lower()
    if not text == 'no text':
        tokens = TweetTokenizer(preserve_case=True).tokenize(text)
        tokens_without_stopwords = [x for x in tokens if x not in deu_stops]
        corpus.append(tokens_without_stopwords)
num_features = 300
min_word_count = 3
num_workers = 2
window_size = 1
subsampling = 1e-3
model = Word2Vec(
corpus,
workers=num_workers,
size=num_features,
min_count=min_word_count,
window=window_size,
sample=subsampling)
model.init_sims(replace=True)
model_name = "billigeplaetze_specific_word2vec_model"
model.save(model_name)
model.most_similar('penis')
wv = KeyedVectors.load(model_name)
len(model.wv.vocab)
print(model.wv.distance('leon','dude'))
print(model.wv.distance('cem','dude'))
model.wv.most_similar('penis')
vocab = list(model.wv.vocab)
X = model[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y']).sample(1000)
plt.rcParams["figure.figsize"] = (20,20)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word, pos)