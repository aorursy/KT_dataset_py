import pandas as pd
reviews = pd.read_csv('../input/olist_order_reviews_dataset.csv')
reviews['num_letters'] = reviews.review_comment_message.str.count('[a-zA-Z]')
reviews = reviews[reviews.num_letters > 0]  # Filter valid comments
reviews.head(3)
import re
import numpy as np
# Nlp
import nltk
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.manifold import TSNE
# Plots
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools, figure_factory
init_notebook_mode(connected=True)
# Wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_wordcloud(text, stopwords, mask=None, max_words=200, max_font_size=100,
                   title=None, title_size=40, image_color=False):

    figure_size = (24, 16)
    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=1200, 
                    height=300,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud);
    plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  

stopwords = nltk.corpus.stopwords.words('portuguese')
comments = reviews.review_comment_message.values
plot_wordcloud(comments, stopwords, title="")
from collections import defaultdict

def generate_ngrams(text, stopwords, n_gram=1):
    token = [w.lower() for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]
    # Remove stopwords and ponctuation
    token = [t for t in token if re.search('[a-zA-Z]', t) and t not in stopwords]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Count words
freq_dict = defaultdict(int)
for sent in comments:
    for word in generate_ngrams(sent, stopwords):
        freq_dict[word] += 1
wdf = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1],
                   columns=["word", "word_count"])

# Count Trigrams
freq_dict = defaultdict(int)
for sent in comments:
    for word in generate_ngrams(sent, stopwords, 3):
        freq_dict[word] += 1
tdf = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1],
                   columns=["trigram", "trigram_count"])

# Sort and filter top 12
tdf = tdf.sort_values(by='trigram_count', ascending=False).iloc[:12]
wdf = wdf.sort_values(by='word_count', ascending=False).iloc[:12]

trace0 = go.Bar(
    y=wdf.word.values,
    x=wdf.word_count.values,
    name='Number of words',
    orientation='h',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    y=tdf.trigram.values,
    x=tdf.trigram_count.values,
    name='Number of trigrams',
    orientation='h',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(
    height=600, width=800,
    title='Words and trigrams in reviews',
    margin=dict(l=150, r=10, t=100, b=100),
    legend=dict(orientation="h")
)
fig['layout']['xaxis1'].update(domain=[0, 0.40])
fig['layout']['xaxis2'].update(domain=[0.6, 1])
iplot(fig)
length = [len(text) for text in comments]
num_letters = [len(re.findall(r'[a-zA-Z]', text)) for text in comments]
num_commas = [text.count(',') for text in comments]
num_dots = [text.count('.') for text in comments]

fig, axis = plt.subplots(1, 2, figsize=(12,4))
pl0 = sns.kdeplot(length, color='navy', label='Review length', ax=axis[0])
pl1 = sns.kdeplot(num_letters, color='orange', label='Number of letters', ax=axis[0])
pl2 = sns.kdeplot(num_dots, color='navy', label='Number of dots', ax=axis[1])
pl3 = sns.kdeplot(num_commas, color='orange', label='Number of commas', ax=axis[1])
review_count = reviews.review_score.value_counts()
trace = go.Bar(x=review_count.index, y=review_count.values)
layout = go.Layout(title='Review scores distribution', height=360, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key."""
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_):
        return data_[self.key].values
class RemoveStopwords(BaseEstimator, TransformerMixin):
    """Remove stopwords from list of tokens list."""
    def fit(self, x, y=None):
        return self

    def transform(self, comments):
        stopwords = nltk.corpus.stopwords.words('portuguese')
        stopwords.extend(['é', 'ok', 'ta', 'tá', 'att', 'att.', 'sr', 'porém',
                          'produto', 'recomendo'])
        return [[tk for tk in tokens if tk not in stopwords] for tokens in comments]
class CorrectSpelling(BaseEstimator, TransformerMixin):
    """Fix a few spelling mistakes in tokens."""
    def fit(self, x, y=None):
        return self

    def transform(self, comments):
        mistakes_dict = {
            'decpcionou': 'decepcionou', 'tô': 'estou', 'to': 'estou',
            'q': 'que', 'pq': 'porque', 'mt': 'muito', 'muiiita': 'muita',
            'estaav': 'estava', 'acabento': 'acabamento', 'orrivel': 'horrível',
            'sertões': 'certos', 'vcs': 'vocês', 'msg': 'mensagem', 'dta': 'data',
            'ñ': 'não', 'n': 'não', 'grates': 'grátis', 'testa-lo': 'testar',
            'superandoo': 'superando', 'atentimento': 'atendimento',
            'cancelacem': 'cancelassem', 'msm': 'mesmo', 'protudo': 'produto',
            'decrarar': 'declarar', 'trasporte': 'transporte', 'decpsionei': 'decepcionei',
            'empuerada': 'empoeirada', 'recebie': 'recebi', 'superr': 'super',
            'nao': 'não', 'mto': 'muito', 'tb': 'também', 'execelente': 'excelente',
            'tao': 'tão', 'blz': 'beleza'
        }
        return [[mistakes_dict[tk] if tk in mistakes_dict else tk for tk in tokens]
               for tokens in comments]
class Stemmer(BaseEstimator, TransformerMixin):
    """Used to reduce words with the portuguese Snowball stemmer."""
    def fit(self, x, y=None):
        return self

    def transform(self, comments):
        #stemmer_ = nltk.stem.snowball.SnowballStemmer('portuguese')
        stemmer_ = nltk.stem.RSLPStemmer()
        return [[stemmer_.stem(tk) for tk in tokens] for tokens in comments]
class Tokenize(BaseEstimator, TransformerMixin):
    """Class to tokenize comments."""
    def fit(self, x, y=None):
        return self

    def transform(self, comments):
        tokenized_comments = list()
        for text in comments:
            # Tokenize and lower
            tokens = [w.lower() for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]
            # Filter tokens that doesn't have letters
            tokens = [t for t in tokens if re.search('[a-zA-Z]', t)]
            tokenized_comments.append(tokens)
        return tokenized_comments
def print_top_words_for_topic(components, feature_names, num_words):
    """Print top words for each topic in components vector."""
    for topic_idx, topic in enumerate(components):
        message = "Topic #%d: " % (topic_idx + 1)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]])
        print(message)

num_topics = 3
rnd_state = 125
lsa_pipeline = Pipeline([
    ('selector', ItemSelector(key='review_comment_message')),
    ('tokenize', Tokenize()),
    ('spelling', CorrectSpelling()),
    ('stopwords', RemoveStopwords()),
    ('stemming', Stemmer()),
    ('tfidf', TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x,
                              analyzer='word', min_df=2, max_df=0.95)),
    ('reduce', TruncatedSVD(n_components=num_topics, random_state=rnd_state))
])

lsa_matrix = lsa_pipeline.fit_transform(reviews)
# Get words and components from pipeline transformers
feat_names = lsa_pipeline.get_params()['tfidf'].get_feature_names()
components = lsa_pipeline.get_params()['reduce'].components_
# Print the 8 most important words for each topic
print_top_words_for_topic(components, feat_names, 8)
plsa_pipeline = Pipeline([
    ('selector', ItemSelector(key='review_comment_message')),
    ('tokenize', Tokenize()),
    ('spelling', CorrectSpelling()),
    ('stopwords', RemoveStopwords()),
    ('stemming', Stemmer()),
    ('tfidf', TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x,
                              analyzer='word', min_df=2, max_df=0.95)),
    ('reduce', NMF(n_components=num_topics, random_state=rnd_state, solver='mu',
                   beta_loss='kullback-leibler', alpha=0.1, l1_ratio=0.5))
])

plsa_matrix = plsa_pipeline.fit_transform(reviews)
# Get words and components from pipeline transformers
feat_names = plsa_pipeline.get_params()['tfidf'].get_feature_names()
components = plsa_pipeline.get_params()['reduce'].components_
# Print the 8 most important words for each topic
print_top_words_for_topic(components, feat_names, 8)
lda_pipeline = Pipeline([
    ('selector', ItemSelector(key='review_comment_message')),
    ('tokenize', Tokenize()),
    ('spelling', CorrectSpelling()),
    ('stopwords', RemoveStopwords()),
    ('stemming', Stemmer()),
    ('countvec', CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x,
                                 analyzer='word', min_df=2, max_df=0.95)),
    ('reduce', LatentDirichletAllocation(n_components=num_topics,
                                         random_state=rnd_state))
])

lda_matrix = lda_pipeline.fit_transform(reviews)
# Get words and components from pipeline transformers
feat_names = lda_pipeline.get_params()['countvec'].get_feature_names()
components = lda_pipeline.get_params()['reduce'].components_
# Print the 8 most important words for each topic
print_top_words_for_topic(components, feat_names, 8)
def scatter_plot(arr, tsne_arr, threshold=0):
    idx = np.amax(arr, axis = 1) >= threshold
    colors = [row.argmax() for row in arr[idx]]
    trace = go.Scattergl(x=tsne_arr[idx, 0], y=tsne_arr[idx, 1],
                         mode='markers', marker=dict(color=colors))
    iplot([trace])
lr = 150
# LSA
lsa_tsne = TSNE(n_components=2, learning_rate=lr).fit_transform(lsa_matrix)
scatter_plot(lsa_matrix, lsa_tsne)

# pLSA
plsa_tsne = TSNE(n_components=2, learning_rate=lr).fit_transform(plsa_matrix)
scatter_plot(plsa_matrix, plsa_tsne)

# LDA
lda_tsne = TSNE(n_components=2, learning_rate=lr).fit_transform(lda_matrix)
scatter_plot(lda_matrix, lda_tsne, threshold=0.5)