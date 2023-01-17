import numpy as np

import pandas as pd

import re, nltk, gensim

import requests

import json

from sklearn.externals import joblib



# Sklearn

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from pprint import pprint



# Plotting tools

import pyLDAvis

import pyLDAvis.sklearn

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
filename = "../input/brazilian-music-samba-lyrics/samba_dataset.csv"

df = pd.read_csv(filename, sep="|")

df.head(3)
# cada letra de samba é um documento

data = [lyrics for lyrics in df.letra] 

print("Temos %d documentos." %len(data))
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))



print(data_words[:1])
def removeStops(texts, stopwords):

    texts_out = []

    for sent in texts:

        texts_out.append(" ".join([token for token in sent if token not in stopwords]))

    return texts_out





stopwords = nltk.corpus.stopwords.words('portuguese')

stopwords += ["nao", "so", "pra", "pro", "pras", "pros"]

# Do lemmatization keeping only Noun, Adj, Verb, Adverb

data_without_stops = removeStops(data_words, stopwords)



# sem stopwords

print(data_without_stops[:2])
vectorizer = CountVectorizer(analyzer='word',       

                             min_df=10,                        # minimum reqd occurences of a word 

                             # stop_words='english',             # remove stop words

                             lowercase=True,                   # convert all words to lowercase

                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3

                             # max_features=50000,             # max number of uniq words

                            )



# data_vectorized = vectorizer.fit_transform(data_lemmatized)

data_vectorized = vectorizer.fit_transform(data_without_stops)
# Materialize the sparse data

data_dense = data_vectorized.todense()



# Compute Sparsicity = Percentage of Non-Zero cells

print("Sparsicity: ", round(((data_dense > 0).sum()/data_dense.size)*100, 2), "%")
lda_model = LatentDirichletAllocation(n_components=5,              

                                      max_iter=10,               

                                      learning_method='online',   

                                      random_state=100,          

                                      batch_size=128,            

                                      evaluate_every = -1,       

                                      n_jobs = -1             

                                     )

lda_output = lda_model.fit_transform(data_vectorized)



print(lda_model)
# Probabilidade logaritmica: quanto maior melhor

print("probabilidade logaritmica: ", round(lda_model.score(data_vectorized), 2))



# Perplexidade: menor melhor.  exp(-1. * log-Probabilidade logaritmica por palavra)

print("Perplexidade: ", round(lda_model.perplexity(data_vectorized), 2))



print("Parâmetros:")

pprint(lda_model.get_params())
# Define Search Param

search_params = {'n_components': [5, 10, 15], 'learning_decay': [.5, .7, .9]}



# Init the Model

lda = LatentDirichletAllocation()



# Init Grid Search Class

model = GridSearchCV(lda, param_grid=search_params)



# Do the Grid Search

model.fit(data_vectorized)
# Melhor modelo

best_lda_model = model.best_estimator_



# Hiperparâmetros do modelo

print("Melhores parâmetros: ", model.best_params_)



# probabilidade logarítmica

print("Melhor score de probabilidade logarítmica: ", model.best_score_)



# Perplexidade

print("Perplexidade do modelo: ", best_lda_model.perplexity(data_vectorized))
results = pd.DataFrame(model.cv_results_)



current_palette = sns.color_palette("Set2", 3)



plt.figure(figsize=(12,8))



sns.lineplot(data=results,

             x='param_n_components',

             y='mean_test_score',

             hue='param_learning_decay',

             palette=current_palette,

             marker='o'

            )



plt.show()
# Create Document - Topic Matrix

lda_output = best_lda_model.transform(data_vectorized)



# column names

topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]



# index names

docnames = ["Doc" + str(i) for i in range(len(data))]



# Make the pandas dataframe

df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)



# Get dominant topic for each document

dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic['dominant_topic'] = dominant_topic



# Styling

def color_green(val):

    color = 'green' if val > .1 else 'black'

    return 'color: {col}'.format(col=color)



def make_bold(val):

    weight = 700 if val > .1 else 400

    return 'font-weight: {weight}'.format(weight=weight)



# Apply Style

df_document_topics = df_document_topic.style.applymap(color_green).applymap(make_bold)

df_document_topics_first10 = df_document_topic[:10].style.applymap(color_green).applymap(make_bold)

df_document_topics_first10
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")

df_topic_distribution.columns = ['Topic Num', 'Num Documents']

df_topic_distribution
vocab = vectorizer.get_feature_names()



# data_vectorized

topic_words = {}

n_top_words = 5



for topic, comp in enumerate(best_lda_model.components_):

    # for the n-dimensional array "arr":

    # argsort() returns a ranked n-dimensional array of arr, call it "ranked_array"

    # which contains the indices that would sort arr in a descending fashion

    # for the ith element in ranked_array, ranked_array[i] represents the index of the

    # element in arr that should be at the ith index in ranked_array

    # ex. arr = [3,7,1,0,3,6]

    # np.argsort(arr) -> [3, 2, 0, 4, 5, 1]

    # word_idx contains the indices in "topic" of the top num_top_words most relevant

    # to a given topic ... it is sorted ascending to begin with and then reversed (desc. now)    

    word_idx = np.argsort(comp)[::-1][:n_top_words]



    # store the words most relevant to the topic

    topic_words[topic] = [vocab[i] for i in word_idx]

    

for topic, words in topic_words.items():

    print('Topic: %d' % topic)

    print('  %s' % ', '.join(words))
# tranformando objeto style em um dataframe pandas

df2 = pd.DataFrame(data=df_document_topics.data, columns=df_document_topics.columns)

df2.head()
# associando os interpretes aos tópicos 

# dos sambas que eles cantam

df2["artista"] = df["artista"].tolist()

df2.head()
# Artistas que mais aparecem dentro de cada tópico

df2.groupby(["dominant_topic"])['artista'].agg(pd.Series.mode).to_frame()
# os 5 artistas que mais aparecem no

# tópico 0 e quantidade de sambas 

df2[df2["dominant_topic"]==0].groupby(["artista"]).size().sort_values(ascending=False)[:5]
# os 5 artistas que mais aparecem no

# tópico 1 e quantidade de sambas 

df2[df2["dominant_topic"]==1].groupby(["artista"]).size().sort_values(ascending=False)[:5]
# os 5 artistas que mais aparecem no

# tópico 2 e quantidade de sambas 

df2[df2["dominant_topic"]==2].groupby(["artista"]).size().sort_values(ascending=False)[:5]
# os 5 artistas que mais aparecem no

# tópico 3 e quantidade de sambas 

df2[df2["dominant_topic"]==3].groupby(["artista"]).size().sort_values(ascending=False)[:5]
# os 5 artistas que mais aparecem no

# tópico 4 e quantidade de sambas 

df2[df2["dominant_topic"]==4].groupby(["artista"]).size().sort_values(ascending=False)[:5]