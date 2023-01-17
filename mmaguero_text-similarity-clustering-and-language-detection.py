import re, string 

import pandas as pd 

import numpy as np

from time import time  

import re, itertools, random

from collections import defaultdict

import spacy

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.decomposition import TruncatedSVD

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('spanish'))

from gensim.models import Word2Vec

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style("darkgrid")

from scipy.spatial.distance import cdist

# uncomment for first run, before put on Internet kernel (settings)

!python -m spacy download es_core_news_md

!python -m spacy link es_core_news_md es_md

nlp = spacy.load('es_md', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
df = pd.read_csv("../input/tweets-municipalidad-asuncion/tweets_municipalidad.csv")
df.head(10)
def clean_text(text):

    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''

    if text is None:

        return ''

    text = str(text).replace("nan",'').lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    # Remove a sentence if it is only one word long

    if len(text) >= 2:

        return ' '.join(word for word in text.split() if word not in STOPWORDS)



df_clean = pd.DataFrame(df.tweet.apply(lambda x: clean_text(x)))

df_clean = df_clean.dropna()

df_clean = df_clean.reset_index(drop=True)
df_clean.head(10)
def lemmatizer(text):        

    sent = []

    doc = nlp(text)

    for word in doc:

        sent.append(word.lemma_)

    return " ".join(sent)



df_clean["text_lemmatize"] = df_clean.apply(lambda x: lemmatizer(x['tweet']), axis=1)
df_clean.head(10)
df_clean['text_lemmatize_clean'] = df_clean['text_lemmatize'].str.replace('-PRON-', '')
sentences = [row.split() for row in df_clean['text_lemmatize_clean']]

word_freq = defaultdict(int)

for sent in sentences:

    for i in sent:

        word_freq[i] += 1

len(word_freq)
sorted(word_freq, key=word_freq.get, reverse=True)[:10]
# min_count: minimum number of occurrences of a word in the corpus to be included in the model.

# window: the maximum distance between the current and predicted word within a sentence.

# size: the dimensionality of the feature vectors

# workers: I know kaggle system is having 4 cores without gpu and 2 with gpu, 

w2v_model = Word2Vec(min_count=100,

                     window=3,

                     size=200,

                     workers=4)
# this line of code to prepare the model vocabulary

w2v_model.build_vocab(sentences)
# train word vectors

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
len(w2v_model.wv.vocab)
# As we do not plan to train the model any further, 

# we are calling init_sims(), which will make the model much more memory-efficient

w2v_model.init_sims(replace=True)
'dengue' in w2v_model.wv.vocab
w2v_model.wv.most_similar(positive=['dengue','mosquito','criadero'])

#w2v_model.wv.most_similar(negative=['dengue','mosquito','criadero','minga'])
# how similar are these two words to each other 

w2v_model.wv.similarity('mosquito','dengue')
w2v_model.wv.doesnt_match(['dengue','mosquito','criadero','minga'])
w2v_model.wv.most_similar(positive=['cateura','dengue'], negative=['mburicao'], topn=3)
def tsne_plot(model, perplexity=10, n_iter=1000):

    "Create TSNE model and plot it"

    labels = []

    tokens = []



    i = 0

    for word in sorted(model.wv.vocab.keys(), reverse=True):

        tokens.append(model[word])

        labels.append(word)

        i+=1

        if i >= 499:

            break

        

    tsne_model = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, n_iter=n_iter)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

    

    x_min, x_max = np.min(new_values, 0), np.max(new_values, 0)

    X = (new_values - x_min) / (x_max - x_min)

    shown_images = np.array([[1., 1.]])  # just something big

    

    plt.figure(figsize=(20, 20)) 

    for i in range(len(x)):

        dist = np.sum((X[i] - shown_images) ** 2, 1)

        '''if np.min(dist) < 1e-3:

            # don't show points that are too close

            continue'''

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(3, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
# Use t-SNE to represent high-dimensional data 

# and the underlying relationships between vectors in a lower-dimensional space.

tsne_plot(w2v_model,40,5000)
# First get the embeddings into a matrix

embedding_size=200

embeddings = np.zeros((len(w2v_model.wv.index2word), embedding_size))

for i in range(0, len(w2v_model.wv.index2word)):

    w = w2v_model.wv.index2word[i]

    embeddings[i] = w2v_model.wv[w]
svd = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=500, random_state=101)

embeddings_2d_projection = svd.fit_transform(embeddings)
# Train a K-means cluster model with 6 clusters

n_clusters = 6

embedding_cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
centroid_embedding_nearest_words = []

for centroid_embedding in embedding_cluster_model.cluster_centers_:

    centroid_embedding_nearest_words.append(

        np.argsort([i[0] for i in cdist(embeddings, np.array([centroid_embedding]), "euclidean")])[0:10]

    )
plt.figure(figsize=(10,10))

colors = itertools.cycle(["b","g","r","c","m","y","k","w"])

c = 0

for word_indices in centroid_embedding_nearest_words:

    clr = next(colors)

    plt.scatter(

        embeddings_2d_projection[word_indices,0],

        embeddings_2d_projection[word_indices,1],

        color=clr,

        label="Cluster " + str(c)

    )

    for ix in word_indices:

        x, y = embeddings_2d_projection[ix,:]

        plt.annotate(w2v_model.wv.index2word[ix], (x, y))

    c+=1

plt.legend(loc='lower left')

plt.show()
def tsnescatterplot(model, word, list_names):

    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,

    its list of most similar words, and a list of words.

    """

    arrays = np.empty((0, 200), dtype='f')

    word_labels = [word]

    color_list  = ['red']



    # adds the vector of the query word

    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    

    # gets list of most similar words

    close_words = model.wv.most_similar([word])

    

    # adds the vector for each of the closest words to the array

    for wrd_score in close_words:

        wrd_vector = model.wv.__getitem__([wrd_score[0]])

        word_labels.append(wrd_score[0])

        color_list.append('blue')

        arrays = np.append(arrays, wrd_vector, axis=0)

    

    # adds the vector for each of the words from list_names to the array

    for wrd in list_names:

        wrd_vector = model.wv.__getitem__([wrd])

        word_labels.append(wrd)

        color_list.append('green')

        arrays = np.append(arrays, wrd_vector, axis=0)

        

    # Reduces the dimensionality from 200 to 12 dimensions with PCA

    reduc = PCA(n_components=12).fit_transform(arrays)

    

    # Finds t-SNE coordinates for 2 dimensions

    np.set_printoptions(suppress=True)

    

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    

    # Sets everything up to plot

    df = pd.DataFrame({'x': [x for x in Y[:, 0]],

                       'y': [y for y in Y[:, 1]],

                       'words': word_labels,

                       'color': color_list})

    

    fig, _ = plt.subplots()

    fig.set_size_inches(9, 9)

    

    # Basic plot

    p1 = sns.regplot(data=df,

                     x="x",

                     y="y",

                     fit_reg=False,

                     marker="o",

                     scatter_kws={'s': 40,

                                  'facecolors': df['color']

                                 }

                    )

    

    # Adds annotations one by one with a loop

    for line in range(0, df.shape[0]):

         p1.text(df["x"][line],

                 df['y'][line],

                 '  ' + df["words"][line].title(),

                 horizontalalignment='left',

                 verticalalignment='bottom', size='medium',

                 color=df['color'][line],

                 weight='normal'

                ).set_size(15)



    

    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)

    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)

            

    plt.title('t-SNE visualization for {}'.format(word.title()))
tsnescatterplot(w2v_model, 'dengue', sorted(word_freq, key=word_freq.get, reverse=True)[:10])
tsnescatterplot(w2v_model, 'dengue', ['mburicao','cateura'])
tsnescatterplot(w2v_model, 'dengue', [i[0] for i in w2v_model.wv.most_similar(negative=['dengue'])])
tsnescatterplot(w2v_model, "dengue", [t[0] for t in w2v_model.wv.most_similar(positive=["dengue"], topn=20)][10:])
import fasttext

!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

model = fasttext.load_model('lid.176.bin')

import langid
def set_language(text):

    lang = 'unknown'

    try:

        lang1 = model.predict(text, k=2) # against two languages

        lang1 = lang1[0][0].replace('__label__','') if lang1[0][0].replace('__label__','') in ['en','es','gn'] else lang1[0][0].replace('__label__','') if lang1[1][0]>=0.7 else 'undefined' #fasttext            

    except:

        lang1 = lang 

        

    # priority fasttext and es en gn

    if text: #example -> lang1:es, lang2:pt

        if (lang1=='gn' or lang1=='es' or lang1=='en'):

            return lang1

        else:

            try:

                lang2 = langid.classify(text)[0] 

            except:

                lang2 = lang

            if (lang2=='gn' or lang2=='es' or lang2=='en'):

                return lang2

            elif (lang1==lang2):

                return lang1

            else:

                return lang1
set_language('Tapeuahẽporãite Vikipetãme') # -> https://gn.wikipedia.org/wiki/Ape
df_clean.loc[4,'language'] = ""



df_clean["language"]=df_clean["tweet"].apply(lambda text: set_language(text))
plt.figure(num=None, figsize=(20, 16), dpi=300, facecolor='w', edgecolor='k')

ax = sns.countplot(y="language", data=df_clean)

ax = ax.set_title('Languages count')

plt.show()
for index, row in df_clean.iterrows():

    if(row['language']=='gn'):

        print(row['tweet'], row['language'])
for index, row in df_clean.iterrows():

    if(row['language'] in ['unknown','undefined']):

        if('__label__gn' in model.predict(row['tweet'], k=10)[0]):

            print(row['tweet'], row['language'])