# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re  # For preprocessing

import pandas as pd  # For data handling

from time import time  # To time our operations

from collections import defaultdict  # For word frequency



import spacy  # For preprocessing



import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
data = pd.read_csv("../input/southparklines/All-seasons.csv")

data.head()

# data.info()
# Remove column "Season" 

data = data.drop(['Season'], axis = 1)

# Remove column "Episode" 

data = data.drop(['Episode'], axis = 1)

data.head()
data.shape
# Checking for missing values

data.isnull().sum()
from IPython.display import Image

# load english language model. Disable Named Entity Recognition ('ner') and 'parser' in Natural Language Processing (nlp) for speed (check the image)

nlp = spacy.load('en', disable=['ner', 'parser'])



def cleaning(doc):

    # Lemmatizes and removes stopwords

    # doc needs to be a spacy Doc object

    # token.lemma_ is the base form of the word (for example: token.text_= APPLE token.lemma_= apple)

    # token.is_stop is a boolean value that represent if the word is one of the most common words on the language(for example: "for", "is"..) 

    txt = [token.lemma_ for token in doc if not token.is_stop]

    ''' print the different parameter for token in doc

        for token in doc:

            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

    '''

    # Word2Vec uses context words to learn the vector representation of a target word,

    # if a sentence is only one or two words long,

    # the benefit for the training is very small

    if len(txt) > 2:

        return ' '.join(txt)
# Removes non-alphabetic characters

brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in data['Line'])

# Use spaCy.pipe() attribute to speed-up the cleaning process

t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

# print(txt)

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
# New DataFrame contains data without duplicates in one column named "Clean"

data_clean = pd.DataFrame({'Clean': txt})

data_clean = data_clean.dropna().drop_duplicates()

data_clean.shape
data_clean.head()
# Detect bigrams (common phrases) from a list of sentences. For example 'mrs_garrison'

from gensim.models.phrases import Phrases, Phraser

# As Phrases() takes a list of list of words as input

sent = [row.split() for row in data_clean['Clean']]

# Creates the relevant phrases from the list of sentences

phrases = Phrases(sent, min_count=30, progress_per=10000)

# Export the trained model = use less RAM, faster processing

bigram = Phraser(phrases)

# Transform the corpus based on the bigrams detected

sentences = bigram[sent]
# The "defaultdict" will simply create any items that you try to access (provided of course they do not exist yet).

# This is useful to avoid that Python dictionary throws a KeyError if you try to get an item with a key that is not currently in the dictionary.

word_freq = defaultdict(int)

for sent in sentences:

    for i in sent:

        word_freq[i] += 1

len(word_freq)
sorted(word_freq, key=word_freq.get, reverse=True)[:10]
import multiprocessing



from gensim.models import Word2Vec
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=20,

                     window=2,

                     size=300,

                     sample=6e-5, 

                     alpha=0.03, 

                     min_alpha=0.0007, 

                     negative=20,

                     workers=cores-1)
t = time()



w2v_model.build_vocab(sentences, progress_per=10000)



print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
t = time()



w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)



print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.wv.most_similar(positive=["eric"])
w2v_model.wv.most_similar(positive=["eric_cartman"])
w2v_model.wv.most_similar(positive=["kenny"])
w2v_model.wv.most_similar(positive=["chef"])
w2v_model.wv.similarity("chef", 'singer')
w2v_model.wv.similarity("kyle", 'jewish')
w2v_model.wv.doesnt_match(['chef', 'token_black', 'stanley'])
w2v_model.wv.doesnt_match(['liane', 'sheila', 'bebe'])
w2v_model.wv.most_similar(positive=["bebe", "popular"], negative=["chef"], topn=3)
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

 

import seaborn as sns

sns.set_style("darkgrid")



from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
def tsnescatterplot(model, word, list_names):

    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,

    its list of most similar words, and a list of words.

    """

    arrays = np.empty((0, 300), dtype='f')

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

        

    # Reduces the dimensionality from 300 to 19 dimensions with PCA

    reduc = PCA(n_components=19).fit_transform(arrays)

    

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

    
tsnescatterplot(w2v_model, 'eric', ['dog', 'bird', 'ah', 'kill', 'bob', 'hat', 'drink', 'bebe'])