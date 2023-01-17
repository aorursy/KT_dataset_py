import re
import multiprocessing
from tqdm import tqdm

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from scipy.stats import norm
from gensim.models import word2vec
from kaggle.competitions import twosigmanews

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

%matplotlib inline
cpu_count = 2*multiprocessing.cpu_count()-1
print('Number of CPUs: {}'.format(cpu_count))
env = twosigmanews.make_env()
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
assetName = 'Companhia de Bebidas das Americas Ambev'
df = market_train_df[market_train_df['assetName']==assetName]
df_news = news_train_df[news_train_df['assetName'] == assetName]
df.head()
data_from = '2012-03-10'
data_to = '2014-09-10'

df_small = df[ (df['time']>data_from) & (df['time'] < data_to)]
df_news_small = df_news[(df_news['time'] > data_from) & (df_news['time'] < data_to)]
plt.figure(figsize=(16,5))

#plt.subplot(2,1,1)
plt.plot(df_small['time'], df_small['close'])
plt.title(assetName)
plt.xlabel('Time', fontsize=16)
'''
plt.subplot(2,1,2)
plt.plot(df_news_small['time'], df_news_small['sentimentNegative'], '--ro', label='Neg')
plt.plot(df_news_small['time'], df_news_small['sentimentNeutral'], '--bo', label='Neu')
plt.plot(df_news_small['time'], df_news_small['sentimentPositive'], '--go', label='Pos')
plt.legend()
'''
plt.figure(figsize=(20,8))
pd.value_counts(df_news_small['headlineTag']).plot(kind="barh")
df_news_small['headline'].head()
def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white',
                              stopwords=STOPWORDS,
                              width=3000,
                              height=2500,
                              max_words=200,
                              random_state=42
                         ).generate(str(corpus))
    return wordCloud
def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus
corpus = get_corpus(df_news_small['headline'].values)
procWordCloud = get_wordCloud(corpus)
df_news_small['headline'].head()
fig = plt.figure(figsize=(20, 8))
plt.subplot(1,2,1)
plt.imshow(procWordCloud)
plt.axis('off')

plt.subplot(1,2,2)

words_count = [len(x.split(' ')) for x in df_news_small['headline'].values]
sns.distplot(words_count,hist=True, kde=False, bins=10, fit=norm)
plt.title("Distribution of words in headline news")
plt.xlabel('Number of words in headline news')
#from:https://github.com/RenatoBMLR/nlpPy/tree/master/src

class TextDataset():

    def __init__(self, df, lang = 'english'):

        self.data = df

        self.tokenizer = TweetTokenizer()
        self.stop_words = set(stopwords.words(lang))
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()
        
    def _get_tokens(self, words):    
        return [word.lower() for word in words.split()]
    
    def _removeStopwords(self, words):
        # Removing all the stopwords
        return [word for word in words if word not in self.stop_words]

    def _removePonctuation(self, words):
        return re.sub(r'[^\w\s]', '', words)

    def _lemmatizing(self, words):
        #Lemmatizing
        return [self.lemmatizer.lemmatize(word) for word in words]

    def _stemming(self, words):
        #Stemming
        return [self.ps.stem(word) for word in words]


    def process_data(self, col = 'content', remove_pontuation=True, remove_stopw = True, lemmalize = False, stem = False):

        self.data = self.data.drop_duplicates(subset=col, keep="last")
        
        proc_col = col
        if remove_pontuation:
            proc_col = col + '_data'
            self.data[proc_col] = self.data[col].apply(lambda x: self._removePonctuation(x) )
        
        # get tokens of the sentence
        self.data[proc_col] = self.data[proc_col].apply(lambda x: self._get_tokens(x))
        if remove_stopw:
            self.data[proc_col] = self.data[proc_col].apply(lambda x: self._removeStopwords(x)) 
        if lemmalize:
            self.data[proc_col] = self.data[proc_col].apply(lambda x: self._lemmatizing(x) )
        if stem:
            self.data[proc_col] = self.data[proc_col].apply(lambda x: self._stemming(x))

        self.data['nb_words'] = self.data[proc_col].apply(lambda x: len(x))
        self.proc_col = proc_col
        
    def __len__(self):
        return len(self.data)
textDataset = TextDataset(df_news_small)
textDataset.process_data(col='headline')
textDataset.data['headline_data'].head()
X = textDataset.data['headline_data'].values
# Set values for various parameters
num_features = 3    # Word vector dimensionality                      
min_word_count = 2   # Minimum word count                        
num_workers = cpu_count  # Number of threads to run in parallel
context = 3          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


W2Vmodel = word2vec.Word2Vec(workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)


W2Vmodel.build_vocab(X)

W2Vmodel.train(X, \
            total_examples=W2Vmodel.corpus_count, epochs=W2Vmodel.epochs)
def plot_tSNE(model,n_samples = 5000):

    #https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm
    output_notebook()
    fig = bp.figure(plot_width=700, plot_height=600, title="A map of " + str(n_samples) + " word vectors",
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
        x_axis_type=None, y_axis_type=None, min_border=1)


    word_vectors = [model[w] for w in model.wv.vocab.keys()][:n_samples]
    #word_vectors = [token for token in f_matrix_train][0:n_samples]

    tsne_model = TSNE(n_components=2, verbose=1, random_state=23)
    tsne_w2v = tsne_model.fit_transform(word_vectors)

    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    tsne_df['words'] = [k for k in model.wv.vocab.keys()][:n_samples]

    fig.scatter(x='x', y='y', source=tsne_df)
    hover = fig.select(dict(type=HoverTool))
    hover.tooltips={"word": "@words"}
    show(fig)
plot_tSNE(W2Vmodel)
