## All minimum setup and libraries required

import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

## TENSORFLOW        
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer   ## Generate dictionary of word encodings
from tensorflow.keras.preprocessing.sequence import pad_sequences

## GENSIM and NLTK
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
nltk.download('wordnet')

# SKLEARN
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

print("Tensorflow\t-\t",tf.__version__)
print("NLTK\t\t-\t",nltk.__version__)
print("Gensim\t\t-\t",nltk.__version__)
path = "../input/medium-articles-with-content/Medium_AggregatedData.csv"
dataframe_full = pd.read_csv(path)
dataframe_imp = pd.read_csv(path)
print("Dataset have been read")
dataframe_full.head()
x = dataframe_full['name'][10]
y = dataframe_full['publicationdescription'][15]
print(x)
print(y)
print(dataframe_full.shape)
print(dataframe_full['name'][10])
print(dataframe_full['name'][11])
print(dataframe_full['name'][12])
print(dataframe_full['title'][10])
print(dataframe_full.columns)
required_col = ['language','subTitle','tagsCount','text','title','url','wordCount','publicationdescription'
               ,'tag_name','name']
most_imp_col = ['subTitle','text','title']
# article_titles = dataframe_full['title']
# art_grp_1 = article_titles[16:25]
# print(art_grp_1)
print(dataframe_full.language.unique())
## Number of rows english rows

english_titles = dataframe_full[dataframe_full['language'] == 'en']
# english_titles.head()
print(english_titles.shape)
## Number of rows dropped after removing null value rows

print(dataframe_imp.shape)
dataframe_imp.dropna(how = 'all')
print(dataframe_imp.shape)
## After dropping non-english and columns that are not required really

dataframe_imp.drop(dataframe_imp[dataframe_imp['language'] != 'en'].index, inplace = True)

dataframe_imp = dataframe_imp.drop(['audioVersionDurationSec', 'codeBlock', 'codeBlockCount',
       'collectionId', 'createdDate', 'createdDatetime', 'firstPublishedDate',
       'firstPublishedDatetime', 'imageCount', 'isSubscriptionLocked',
       'language', 'latestPublishedDate', 'latestPublishedDatetime',
       'linksCount', 'postId', 'readingTime', 'recommends',
       'responsesCreatedCount', 'socialRecommendsCount','tagsCount','totalClapCount', 'uniqueSlug',
       'updatedDate', 'updatedDatetime', 'url', 'vote', 'wordCount',
       'publicationdescription', 'publicationdomain',
       'publicationfacebookPageName', 'publicationfollowerCount',
       'publicationname', 'publicationpublicEmail', 'publicationslug',
       'publicationtags', 'publicationtwitterUsername', 'tag_name', 'slug',
       'name', 'postCount', 'author', 'bio', 'userId', 'userName',
       'usersFollowedByCount', 'usersFollowedCount', 'scrappedDate'], axis=1)

dataframe_imp['index'] = dataframe_imp.index

dataframe_imp.shape
## Run these cell to export the new trimmed down dataset

# dataframe_imp.to_csv("medium_dataset.csv",sep=",")
# new_ds_size = os.stat("medium_dataset.csv").st_size
# new_ds_size = new_ds_size / 1000000
# print("New dataset size in MB = ",new_ds_size)
## Keeping 170000 rows greatly reduces the dataset size to around 100MB

very_reduced_dataset = dataframe_imp[:17000]
# very_reduced_dataset.to_csv("very_reduced_dataset.csv",sep=",")
# print("New dataset size in MB = ",os.stat("very_reduced_dataset.csv").st_size / 1000000)

# from IPython.display import FileLink
# FileLink(r'very_reduced_dataset.csv')
dataframe_imp.head()
print(dataframe_imp.title[15])
print(dataframe_imp.subTitle[15])
# print(dataframe_imp.text[15])      ## Text is too huge to be displayed
print(dataframe_imp.index[15])
## Stemmer initialization for english
stemmer = SnowballStemmer("english")
## Functions for lemmatization, removal of Stopwords

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
### Code to check the function

## Run this for faster execution time wiht less acuracy
doc_sample = very_reduced_dataset[very_reduced_dataset['index'] == 1000].values[0][2]

## Run this for slower execution but better accuracy
# doc_sample = dataframe_imp[dataframe_imp['index'] == 1000].values[0][2]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
    
print(dataframe_imp[dataframe_imp['index'] == 1000].values[0][2])
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))
## Use this to reduce the training time at the cost of loss of rows and some accuracy.
title_list = very_reduced_dataset['title'].astype(str)

## Use this for higher model accuracy but very slow operating time. 
# title_list = dataframe_imp['title'].astype(str)   ## using astype(str) eliminates the floting type error
title_list.describe()
## The titles are preprocessed and saved into processd_titles
## The map function applies the preprocess method on each of the list entries

processed_titles = title_list.map(preprocess)
processed_titles[30:40]
## bow --> Bag of Words

bow = gensim.corpora.Dictionary(processed_titles)

## Finding out words with a min_occurance = 10

min_occurance = 10
count = 0
for k, v in bow.iteritems():
    print(k, v)
    count += 1
    if count > min_occurance:    # We can limit the selection based on the frequency
        break
bow.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
print(bow)
bow_corpus = [bow.doc2bow(doc) for doc in processed_titles]
bow_corpus[:10]
## A example of the BOW for the 1000th title

bow_example = bow_corpus[1000]
for i in range(len(bow_example)):
    print("Word {} (\"{}\") appears {} time.".format(bow_example[i][0], 
           bow[bow_example[i][0]], 
           bow_example[i][1]))
from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# from pprint import pprint

for i in corpus_tfidf:
    print(i)
    break
## LDA when the num_topics = 10

# lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=bow, passes=5, workers=3)
lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                       id2word=bow,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
print(lda_model)
# For optimal performance time set the workers as no of CPU cores-1
for idx, topic in lda_model.print_topics(-1):
    print('\nTopic: {} \nWords: {}'.format(idx, topic))
doc_lda = lda_model[bow_corpus]
topics = lda_model.show_topics(formatted=False)
topic_words = dict(topics[1][1])
print("For topic 1, the words are: ",topic_words)
%pylab inline

import pandas as pd
import pickle as pk
from scipy import sparse as sp
## Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 5, figsize=(18,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in dataframe_imp for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 5, figsize=(18,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.08); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary=lda_model.id2word)
vis
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_titles, dictionary=lda_model.id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
# Compute Coherence Score using UMass
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_titles, dictionary=lda_model.id2word, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
def compute_coherence_values(dictionary, corpus, texts, limit,cs_type, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=cs_type)
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=bow, corpus=bow_corpus, texts=processed_titles, start=2, limit=100, step=5,cs_type = 'c_v')
# Show graph
import matplotlib.pyplot as plt
limit=100; start=2; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values, marker='o')
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
model_list, coherence_values = compute_coherence_values(dictionary=bow, corpus=bow_corpus, texts=processed_titles, start=2, limit=100, step=5,cs_type = 'u_mass')
# Show graph
import matplotlib.pyplot as plt
limit=100; start=2; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values,marker = 'o')
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
## supporting function with num_topics = 60

def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=bow,
                                           num_topics=60, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           per_word_topics=True)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_titles, dictionary=lda_model.id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()
import numpy as np
import tqdm
grid = {}
grid['Validation_Set'] = {}
# Topics range
# min_topics = 2
# max_topics = 11
# step_size = 1
# topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
# alpha = list(np.arange(0.01, 1, 0.3))
alpha = np.array([0.05,0.1,0.5,1,5,10])
# alpha.append('symmetric')
# alpha.append('asymmetric')

# Beta parameter
# beta = list(np.arange(0.01, 1, 0.3))
beta = np.array([0.05,0.1,0.5,1,5,10])
# beta.append('symmetric')

# Validation sets
num_of_docs = len(bow_corpus)
corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
               gensim.utils.ClippedCorpus(bow_corpus, int(num_of_docs*0.75)), 
               bow_corpus]
corpus_title = ['75% Corpus', '100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }
k = 45
# Can take a long time to run
# if 1 == 1:
pbar = tqdm.tqdm(total=200, position=0, leave=True)


# iterate through validation corpuses
for i in range(len(corpus_sets)):
    # iterate through alpha values
    for a in alpha:
        # iterare through beta values
        for b in beta:
            # get the coherence score for the given parameters
            cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=lda_model.id2word, 
                                          k=k, a=a, b=b)
            # Save the model results
            model_results['Validation_Set'].append(corpus_title[i])
            model_results['Topics'].append(k)
            model_results['Alpha'].append(a)
            model_results['Beta'].append(b)
            model_results['Coherence'].append(cv)
            pbar.update(1)
pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
pbar.close()
dataframe_results = pd.read_csv("./lda_tuning_results.csv")
dataframe_results
topics = dataframe_results['Topics']
coherence_values = dataframe_results['Coherence']
dataframe_results['Topics'].unique()
optimal = dataframe_results[dataframe_results['Validation_Set'] == '100% Corpus']
max_coherence_val = optimal['Coherence'].max()
optimal = dataframe_results[dataframe_results['Coherence'] == max_coherence_val]
print(optimal)
# print(val_set)
final_lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                             id2word=bow,
                                             num_topics=60,
                                             random_state=100,
                                             chunksize=100,
                                             passes=10,
                                             alpha = 0.5,
                                             eta=0.1, 
                                             per_word_topics=True)
final_coherence_model_lda = CoherenceModel(model=final_lda_model, texts=processed_titles, dictionary=lda_model.id2word, coherence='c_v')
final_coherence_score = final_coherence_model_lda.get_coherence()
print('\nCoherence Score: ',final_coherence_score )
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(final_lda_model, bow_corpus, dictionary=bow)
vis