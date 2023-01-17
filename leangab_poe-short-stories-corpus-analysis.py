%%capture



!pip install contractions

!pip install gensim



import html

import re

import unicodedata

import string

import spacy

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time

import contractions

import numpy as np

import os

import datetime

import seaborn as sns

import multiprocessing

import nltk

import matplotlib.pyplot as plt

import pyLDAvis

import pyLDAvis.gensim  



from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize

from nltk.corpus import wordnet as wn

from nltk import pos_tag

from requests import get

from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import dendrogram, linkage, ward, cut_tree

from gensim import corpora, models

from gensim.models import word2vec

from gensim.models import CoherenceModel

from gensim.models.phrases import Phrases, Phraser

from sklearn.neighbors import NearestNeighbors

from collections import defaultdict

from collections import Counter

from sklearn.manifold import TSNE



pd.set_option('display.max_rows', 10)



nltk.download('stopwords')

spacy.cli.download("en")



sns.set(style="darkgrid")

%matplotlib inline
url = 'http://www.gutenberg.org/files/25525/25525-h/25525-h.htm#2150link2H_4_0003'

response = get(url)

response.encoding = 'utf-8'



text = re.sub(r"\r|\n", "", response.text)

book1 = re.search(r"<h2[^<]*?THE UNPARALLELED ADVENTURES.*She was dead!.*?p>", text).group(0)

book2 = re.search(r"<h2[^<]*?THE PURLOINED LETTER.*unto Eleonora.*?p>", text).group(0)

book3 = re.search(r"<h2[^<]*?LIGEIA.*opinion upon that.*?p>", text).group(0)

book4 = re.search(r"<h2[^<]*?THE DEVIL IN THE BELFRY.*departed friends.*?p>", text).group(0)

book5 = re.search(r"<h2[^<]*?PHILOSOPHY OF FURNITURE.*it would have been Lilies without, roses within.*?p>", text).group(0)



books = book1 + book2 + book3 + book4 + book5

books = html.unescape(books)

books = re.sub(r"\s+", " ", books)
titles = re.findall(r"<h2.*?h2>", books)

titles = [re.sub(r"<.*?>", "", x).strip() for x in titles]

titles = [re.sub(r"Footnotes.*|Notes.*|[.]|\(.*?\)", "",x, flags=re.I) for x in titles]

titles = [x.strip() for x in titles if x != ""]

sort_index = np.argsort(titles)

titles = np.array(titles)[sort_index]

pd.DataFrame({"Titles":titles}).head(5)
stories = books.split("<h2>")

stories = [re.sub(r"^.*</h2>","", x) for x in stories]

stories = [re.sub(r"<.*?>","", x).strip() for x in stories]

stories = [i for i in stories if i != "" and re.match(r"\(\*1\)", i) is None]

stories = np.array(stories)[sort_index]

print("Number of stories : %s, Number of titles : %s" %(len(stories),len(titles)))
tables = pd.read_html("https://en.wikipedia.org/wiki/Edgar_Allan_Poe_bibliography", header=0)

table = tables[2].sort_values(by="Title")

table.Notes =  [re.sub(r'\[.*\]', "", x) for x in table.Notes.values]

table[:5]
titles2 = [x.upper() for x in table.Title.values.tolist()]

titles2 = np.array([re.sub("[^A-Z]", "", x) for x in titles2])

titles1 = np.array([re.sub("[^A-Z]", "", x) for x in titles])

table_info_subs = [True if x in titles1 else False for x in titles2]

subs = np.array([True if x in titles2 else False for x in titles1])

table = table[table_info_subs]

titles2 = titles2[table_info_subs]

sort_index = np.argsort(titles2)

table = table.iloc[sort_index,:]

titles2 = titles2[sort_index]



titles_not_in_wiki = titles[~subs]

stories_not_in_wiki = stories[~subs]

titles = titles[subs]

stories = stories[subs]

sort_index = np.argsort(titles)

titles = np.array(titles)[sort_index]

stories = stories[sort_index]





titles = np.concatenate((titles, titles_not_in_wiki), axis = 0)

stories = np.concatenate((stories, stories_not_in_wiki), axis = 0)

data = pd.DataFrame({"title": titles, "stories":stories})

table = table.reset_index(drop= True)

data = data.join(table, how="left")

data.columns = ["title", "text", "wikipedia_title", "publication_date", "first_published_in", "classification", "notes"]

data.fillna("", inplace = True)

data[:5]
# PHILOSOPHY OF FURNITURE

## https://en.wikipedia.org/wiki/The_Philosophy_of_Furniture

data.at[62,'classification']= 'Essay'

data.at[62,'publication_date']= 'May 1840'



# MAAELZEL’S CHESS-PLAYER

## https://en.wikipedia.org/wiki/Maelzel%27s_Chess_Player

data.at[63,'classification']= 'Essay'

data.at[63,'publication_date']= 'April 1836'



# OLD ENGLISH POETRY

data.at[64,'classification']= 'Essay'

data.at[64,'publication_date']= "?"



# THE BALLOON-HOAX

## https://en.wikipedia.org/wiki/The_Balloon-Hoax

data.at[65,'classification']= 'Hoax / Fiction'

data.at[65,'publication_date']= 'April 13, 1844'



# THE MYSTERY OF MARIE ROGET

## https://en.wikipedia.org/wiki/The_Mystery_of_Marie_Rog%C3%AAt

## taking the date of publication of the first part

data.at[66,'classification']= 'Detective fiction'

data.at[66,'publication_date']= 'November 1842'



# THE POETIC PRINCIPLE

## http://www.thepoeblog.org/the-poetic-principle-a-rich-intellectual-treat/

## taking the date when the the work was known with Poe alive (published posthumously)

data.at[67,'classification']= 'Essay'

data.at[67,'publication_date']= 'August 17, 1849' 



# THE UNPARALLELED ADVENTURES OF ONE HANS PFAAL

## https://en.wikipedia.org/wiki/The_Unparalleled_Adventure_of_One_Hans_Pfaall

data.at[68,'classification']= 'Hoax / Science fiction'

data.at[68,'publication_date']= 'June 1835' 



# X-ING A PARAGRAPH

data.at[69,'classification']= 'Satire'

data.at[69,'publication_date']= 'May 12, 1849' 



# Some fixes 

## The Purloined Letter

## https://en.wikipedia.org/wiki/The_Purloined_Letter

data.at[51,'publication_date']= "December, 1844"



## Silence - a Fable

## https://en.wikipedia.org/wiki/Poems_by_Edgar_Allan_Poe

data.at[23,'publication_date']= "January 4, 1840"



## https://www.goodreads.com/book/show/8498298-why-the-little-frenchman-wears-his-hand-in-a-sling

data.at[60,'publication_date']= "August 17, 1839"



# Eleonora

## Unknown month, published in "the Gift" as "The Pit and the Pendulum"

data.at[6,'publication_date']= "? 1841"



# The Pit and the Pendulum

data.at[48,'publication_date']= "? 1843"
year = [re.sub(r"(.*)(\d{4})", "\\2", x) for x in  data.publication_date.values]

month = [re.sub(r"[^a-zA-Z]", "", x) for x in  data.publication_date.values]

month[6] = "?"

month[48] = "?"

data["normalized_date"] = ["%s %s" %(x, y) for x,y in zip(month,year)]
data["wikipedia_title"] = [re.sub(r"[\"]", "", x) for x in data.iloc[:,2].values]
data.classification = [re.sub(",", "/", x) for x in data.classification]

data.classification = [re.sub(r"(\w)( +)(/)", "\\1/", x) for x in data.classification]

data.classification = [re.sub(r"(/)( +)(\w)", "/\\3", x) for x in data.classification]

data.classification = [re.sub(" +", "_", x) for x in data.classification]



tokens = [x.split("/") for x in data.classification.values]

tokens = [sorted(x) for x in tokens]

tokens = [",".join(x) for x in tokens]



data.classification = tokens



np.unique(tokens)
data[data == ""] = "?"
data.to_csv("preprocessed_data.csv", index=False)

with pd.option_context('display.max_rows', 100):

    display(data)


"""

Text normalization

"""



def text_normalizer(text, stemming = False, lemmatize = True,

                    find_bigrams = True, min_count = 2, min_token_length = 2):

  

    # remove dialog marks

    text = re.sub("—+", " ", text)

    # remove accents

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8') 

 

    # fix contractions

    text = contractions.fix(text) 

  

    # fix case

    text = text.lower() 

  

    # remove extra newlines

    text = re.sub(r'[\r|\n|\r\n]+', '\n', text) 

  

    # remove special characters and digits

    text = re.sub(r'[^a-zA-Z\s]', ' ', text) 

  

    # remove extra whitespace

    text = re.sub(' +', ' ', text)

    

    tokens = word_tokenize(text)

  

    stopwords_list = nltk.corpus.stopwords.words('english')

    stopwords_list = stopwords_list + list(string.punctuation)

    stopwords_list = set(stopwords_list)

    tokens = [token.strip() for token in tokens]

    tokens = [token for token in tokens if token not in stopwords_list]



    if stemming:

        stemmer = SnowballStemmer("english")

        tokens = [stemmer.stem(token) for token in tokens]

    

    if lemmatize:

        tag_map = defaultdict(lambda : wn.NOUN)

        tag_map['J'] = wn.ADJ

        tag_map['V'] = wn.VERB

        tag_map['R'] = wn.ADV

    

        wordnet_lemmatizer = WordNetLemmatizer()

        tokens = [wordnet_lemmatizer.lemmatize(token,  tag_map[tag[0]]) for token,tag in pos_tag(tokens)]



    tokens = [token for token in tokens if len(token) > min_token_length]

  

    return tokens





"""

Computation of Word2vec average vectors

"""



def average_word_vectors(words, model, vocabulary, num_features):

    feature_vector = np.zeros((num_features, ), dtype = "float64")

    nwords = 0.

    for word in words:

        if word in vocabulary:

            nwords = nwords + 1.

            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:

        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector

  

    

def averaged_word_vectorizer(corpus, model, num_features):

    vocabulary = set(model.wv.index2word)

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]

    return np.array(features)



"""

LDA Coherence computation

"""



def compute_coherence_values(model, dictionary, corpus, texts, limit, start=2, step=2):

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        print("Modeling = " + str(num_topics))

        model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, 

                                         passes=20, random_state = 100, 

                                         alpha='auto', eta='auto', update_every=1, chunksize=100)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence = coherencemodel.get_coherence()

        coherence_values.append(coherence)

        print("Coherence: " + str(coherence))

    return model_list, coherence_values





def plot_coherence(coherence_values, start, limit, step):

    x = range(start, limit, step)

    plt.plot(x, coherence_values)

    plt.xlabel("Num Topics")

    plt.ylabel("Coherence score")

    plt.legend(("coherence_values"), loc='best')

    plt.show()

    

    

"""

Genererate a table output with the LDA results

"""

    

def format_topics_sentences(ldamodel, corpus, texts):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row in enumerate(ldamodel[corpus]):

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), 

                                                                  round(prop_topic,4), topic_keywords]), 

                                                       ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)
normalized_stories = [text_normalizer(i) for i in data.text.values]
sns.distplot([len(x) for x in normalized_stories], axlabel="# words")
# Set values for various parameters

feature_size = 300 # Word vector dimensionality

window_context = 20 # Context window size

min_word_count = 3 # Minimum word count

sample = 6e-5 # Downsample setting for frequent words

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

negative = 10

alpha = 0.03

epochs = 300

min_alpha =  0.03 / epochs #alpha - (min_alpha * epochs) ~ 0.00



bigram = Phrases(normalized_stories, min_count=5, delimiter=b' ')

bigram_text = [bigram[line] for line in normalized_stories]



w2v_model = word2vec.Word2Vec(size=feature_size, window=window_context, min_count=min_word_count, 

                              sample=sample, iter=epochs, workers=cores-1, negative = negative, 

                              alpha = alpha, min_alpha = min_alpha)



w2v_model.build_vocab(bigram_text)



w2v_model.train(normalized_stories, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)

w2v_model.corpus_total_words
w2v_feature_array = averaged_word_vectorizer(corpus=normalized_stories, model=w2v_model, num_features=feature_size)

corpus_df = pd.DataFrame(w2v_feature_array)
w2v_model.most_similar(positive=["dupin"])
w2v_model.most_similar(positive=["clock"])
w2v_model.most_similar(positive=["poetry"])
similarity_matrix = cosine_similarity(w2v_feature_array)

similarity_df = pd.DataFrame(similarity_matrix)

similarity_df

Z = linkage(similarity_matrix, 'ward')



labels = data.classification.values



classes=np.unique(labels)

classes = [re.sub(".*Horror.*", "Horror", x) for x in classes]

classes = [re.sub(".*Humor.*", "Humor", x) for x in classes]

classes = [re.sub(".*Parody.*", "Humor", x) for x in classes]

classes = [re.sub(".*Satire.*", "Humor", x) for x in classes]

classes = [re.sub(".*Science fiction.*", "Science fiction", x) for x in classes]

classes = [re.sub(".*Detective fiction.*", "Detective fiction", x) for x in classes]

classes = [re.sub(".*Satire.*", "Satire", x) for x in classes]

classes = np.unique(classes)



labels = [re.sub(".*Horror.*", "Horror", x) for x in labels]

labels = [re.sub(".*Humor.*", "Humor", x) for x in labels]

labels = [re.sub(".*Parody.*", "Humor", x) for x in labels]

labels = [re.sub(".*Satire.*", "Humor", x) for x in labels]

labels = [re.sub(".*Science fiction.*", "Science fiction", x) for x in labels]

labels = [re.sub(".*Detective fiction.*", "Detective fiction", x) for x in labels]

labels = [re.sub(".*Satire.*", "Satire", x) for x in labels]



index= [int(np.where(val == classes)[0]) for i, val in enumerate(labels)] 
w2v_feature_array.shape

tsne_model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=6)

X_2d = tsne_model.fit_transform(w2v_feature_array)



plt.subplots(figsize=(20,10))

from sklearn.preprocessing import MinMaxScaler

X_2d = MinMaxScaler().fit_transform(X_2d)

df= pd.DataFrame({"x":X_2d[:, 0], "y":X_2d[:, 1], "labels":labels, "titles":titles})



markers = ('v', 'o', '^', 'X', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', '<')

ax=sns.scatterplot(data=df, x="x", y="y", hue="labels", style="labels", s=170, markers = markers)

#For each point, we add a text inside the bubble

for line in range(0,df.shape[0]):

     np.random.seed(line + 2)

     ax.text(df.x[line] + 0.01, df.y[line] + np.random.uniform(-0.03, 0.03), 

             df.titles[line], size='small', color='black', weight=540, ha="left", va="top")

ax.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)

plt.show()

info = ["%s   (%s)" %(x,y) for x,y in zip(data.title.values, data.classification.values + " - " + data.normalized_date.values)]

info[:5]


pal=sns.color_palette("dark", n_colors=len(classes))

pal=pal.as_hex()

pal = [pal[i] for i in index]



fig,ax=plt.subplots(figsize=(12, 18))

dd= dendrogram(Z,  labels=info, orientation = "left", leaf_font_size = 10, color_threshold = 4)

pal = [pal[i] for i in dd["leaves"]]

ax = plt.gca()

xlbls = ax.get_ymajorticklabels()

num=0

for lbl in xlbls:

    lbl.set_color(pal[num])

    num+=1



ax.set_title('Hierarchical Clustering Dendrogram')

ax.set_xlabel('sample index')

ax.set_ylabel('distance (Ward)')

plt.show()
x = ward(Z)

clusters = cut_tree(Z, n_clusters=[2, 4, 6])

data[["level_0", "level_1", "level_2"]] = clusters
taged_words = [pos_tag(x, "universal") for x in normalized_stories]

word_type = [Counter([y for x,y in z]) for z in taged_words ]

word_classification = pd.DataFrame(word_type, index = titles)

#word_classification["total"] = word_classification.sum(axis=1)

word_classification.fillna(0, inplace=True)

word_classification = word_classification.apply(lambda x: x/x.sum(), axis = 1)

word_classification 

sns.clustermap(word_classification, standard_scale=1, yticklabels=True, robust=True, metric = "euclidean", method = "ward", cmap="vlag", figsize = (13,13))
# LDA 



# filtering grammatical structure



#VERB - verbs (all tenses and modes)

#NOUN - nouns (common and proper)

#PRON - pronouns

#ADJ - adjectives

#ADV - adverbs

#ADP - adpositions (prepositions and postpositions)

#CONJ - conjunctions

#DET - determiners

#NUM - cardinal numbers

#PRT - particles or other function words

#X - other: foreign words, typos, abbreviations

    

filtered = []

for x in normalized_stories:

    tmp = []

    for y,z in pos_tag(x, "universal"):

        if z not in  ["PRON", "ADP", "CONJ", "DET", "PRT", "X"]:

            tmp.append(y)

    filtered.append(tmp)





# Create a dictionary representation of the documents.

dictionary = corpora.Dictionary(filtered)



# Filter out words that occur less than 5 documents, or more than 50% of the documents.

dictionary.filter_extremes(no_below=5, no_above=0.5)



corpus = [dictionary.doc2bow(text) for text in  filtered]



ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20, random_state = 5,  alpha='auto', eta = 'auto', update_every=1, chunksize=100)
# Can take a long time to run.

model_list, coherence_values = compute_coherence_values(model = ldamodel, dictionary=dictionary, corpus=corpus, texts=filtered, start=2, limit=14, step=1)
# Show graph

plot_coherence(coherence_values, 2, 14, 1)
#optimal_model = model_list[np.argmax(coherence_values)]

optimal_model = model_list[6] # I am choosing K = 8, it is a stable point as shown by the plot, and with an interesting diversity of topics

model_topics = optimal_model.show_topics(formatted=False)

for i in optimal_model.print_topics(num_words=10):

    print(i)
pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(optimal_model, corpus, dictionary)

vis
df_topic = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=info)

data[["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]] = df_topic[["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]]

data.to_csv("final_data.csv", index=False)

data
neigh = NearestNeighbors(n_neighbors=4, metric='cosine')

neigh.fit(w2v_feature_array)

A = neigh.kneighbors_graph(w2v_feature_array).toarray()

np.fill_diagonal(A, 0)
titles = np.array(titles)

out = []

for i in range(len(titles)):

  out.append(np.array(titles)[A[i].astype("bool")])


recommended = pd.DataFrame(out, index= titles)

recommended.sort_index()

recommended = pd.DataFrame(recommended.values, info)

recommended.to_csv("recommended.csv", index=True)

with pd.option_context('display.max_rows', 100):

    display(recommended)
