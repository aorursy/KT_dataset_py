import os

print(os.listdir("../input"))



import re

import numpy as np

import pandas as pd

from pprint import pprint



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel



# spacy for lemmatization

import spacy



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim 

import matplotlib.pyplot as plt

%matplotlib inline



# Enable logging for gensim - optional

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
# NLTK Stop words



from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use','a','about', 'above', 'across'])
st1= ['after', 'afterwards','again','against', 'all', 'almost','alone','along',

           'already',

           'also',

           'although',

           'always',

           'am',

           'among',

           'amongst',

           'amoungst',

           'amount',

           'an',

           'and',

           'another',

           'any',

           'anyhow',

           'anyone',

           'anything',

           'anyway',

           'anywhere',

           'are',

           'around',

           'as',

           'at',

           'back',

           'be',

           'became',

           'because',

           'become',

           'becomes',

           'becoming',

           'been',

           'before',

           'beforehand',

           'behind',

           'being',

           'below',

           'beside',

           'besides',

           'between',

           'beyond',

           'bill',

           'both',

           'bottom',

           'but',

           'by',

           'call',

           'can',

           'cannot',

           'cant',

           'co',

           'con',

           'could',

           'couldnt',

           'cry',

           'de',

           'describe',

           'detail',

           'do',

           'done',

           'down',

           'due',

           'during',

           'each',

           'eg',

           'eight',

           'either',

           'eleven',

           'else',

           'elsewhere',

           'empty',

           'enough',

           'etc',

           'even',

           'ever',

           'every',

           'everyone',

           'everything',

           'everywhere',

           'except',

           'few',

           'fifteen',

           'fifty',

           'fill',

           'find',

           'fire',

           'first',

           'five',

           'for',

           'former',

           'formerly',

           'forty',

           'found',

           'four',

           'from',

           'front',

           'full',

           'further',

           'get',

           'give',

           'go',

           'had',

           'has',

           'hasnt',

           'have',

           'he',

           'hence',

           'her',

           'here',

           'hereafter',

           'hereby',

           'herein',

           'hereupon',

           'hers',

           'herself',

           'him',

           'himself',

           'his',

           'how',

           'however',

           'hundred',

           'i',

           'ie',

           'if',

           'in',

           'inc',

           'indeed',

           'interest',

           'into',

           'is',

           'it',

           'its',

           'itself',

           'keep',

           'last',

           'latter',

           'latterly',

           'least',

           'less',

           'ltd',

           'made',

           'many',

           'may',

           'me',

           'meanwhile',

           'might',

           'mill',

           'mine',

           'more',

           'moreover',

           'most',

           'mostly',

           'move',

           'much',

           'must',

           'my',

           'myself',

           'name',

           'namely',

           'neither',

           'never',

           'nevertheless',

           'next',

           'nine',

           'no',

           'nobody',

           'none',

           'noone',

           'nor',

           'not',

           'nothing',

           'now',

           'nowhere',

           'of',

           'off',

           'often',

           'on',

           'once',

           'one',

           'only',

           'onto',

           'or',

           'other',

           'others',

           'otherwise',

           'our',

           'ours',

           'ourselves',

           'out',

           'over',

           'own',

           'part',

           'per',

           'perhaps',

           'please',

           'put',

           'rather',

           're',

           'same',

           'see',

           'seem',

           'seemed',

           'seeming',

           'seems',

           'serious',

           'several',

           'she',

           'should',

           'show',

           'side',

           'since',

           'sincere',

           'six',

           'sixty',

           'so',

           'some',

           'somehow',

           'someone',

           'something',

           'sometime',

           'sometimes',

           'somewhere',

           'still',

           'such',

           'system',

           'take',

           'ten',

           'than',

           'that',

           'the',

           'their',

           'them',

           'themselves',

           'then',

           'thence',

           'there',

           'thereafter',

           'thereby',

           'therefore',

           'therein',

           'thereupon',

           'these',

           'they',

           'thick',

           'thin',

           'third',

           'this',

           'those',

           'though',

           'three',

           'through',

           'throughout',

           'thru',

           'thus',

           'to',

           'together',

           'too',

           'top',

           'toward',

           'towards',

           'twelve',

           'twenty',

           'two',

           'un',

           'under',

           'until',

           'up',

           'upon',

           'us',

           'very',

           'via',

           'was',

           'we',

           'well',

           'were',

           'what',

           'whatever',

           'when',

           'whence',

           'whenever',

           'where',

           'whereafter',

           'whereas',

           'whereby',

           'wherein',

           'whereupon',

           'wherever',

           'whether',

           'which',

           'while',

           'whither',

           'who',

           'whoever',

           'whole',

           'whom',

           'whose',

           'why',

           'will',

           'with',

           'within',

           'without',

           'would',

           'yet',

           'you',

           'your',

           'yours',

           'yourself',

           'yourselves']
stop_words.extend(st1)
data = pd.read_csv("../input/uci-news-aggregator.csv")
bg = data[data.CATEGORY == 'b']

tg = data[data.CATEGORY == 't']

eg = data[data.CATEGORY == 'e']

mg = data[data.CATEGORY == 'm']
bg_rows = np.random.choice(bg.index.values, 750)

bg_data = bg.ix[bg_rows]



tg_rows = np.random.choice(tg.index.values, 750)

tg_data = tg.ix[tg_rows]



eg_rows = np.random.choice(eg.index.values, 750)

eg_data = eg.ix[eg_rows]



mg_rows = np.random.choice(mg.index.values, 750)

mg_data = mg.ix[mg_rows]
data = bg_data.append([tg_data, eg_data, mg_data])
title = data['TITLE']

category = data['CATEGORY']
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer

 

NUM_TOPICS = 4



# Converting the document to a matrix of token counts



vectorizer = CountVectorizer(min_df=5, max_df=0.9, 

                             stop_words='english', lowercase=True, 

                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

data_vectorized = vectorizer.fit_transform(title)

 

# Build a Latent Semantic Indexing Model using SVD



lsi_model = TruncatedSVD(n_components=NUM_TOPICS)

lsi_Z = lsi_model.fit_transform(data_vectorized)

print(lsi_Z.shape)
def print_topics(model, vectorizer, top_n=10):

    for idx, topic in enumerate(model.components_):

        print("Topic %d:" % (idx))

        print([(vectorizer.get_feature_names()[i], topic[i])

                        for i in topic.argsort()[:-top_n - 1:-1]])

 

print("LSI Model:")

print_topics(lsi_model, vectorizer)

print("=" * 20)
from sklearn.manifold import TSNE

# NLTK

from nltk.tokenize import RegexpTokenizer

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords

import re



# Visualization

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib

%matplotlib inline

import seaborn as sns



# Bokeh

from bokeh.io import output_notebook

from bokeh.plotting import figure, show

from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider

from bokeh.layouts import column

from bokeh.palettes import all_palettes

output_notebook()
import pandas as pd

from bokeh.io import push_notebook, show, output_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource, LabelSet

output_notebook()



svd = TruncatedSVD(n_components=100)

documents_2d = svd.fit_transform(data_vectorized)

 

df = pd.DataFrame(columns=['x', 'y', 'document'])

df['x'], df['y'], df['document'] = documents_2d[:,0], documents_2d[:,1], range(len(data))

 

source = ColumnDataSource(ColumnDataSource.from_df(df))

labels = LabelSet(x="x", y="y", text="document", y_offset=8,

                  text_font_size="8pt", text_color="#555555",

                  source=source, text_align='center')

 

plot = figure(plot_width=600, plot_height=600)

plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)

plot.add_layout(labels)

show(plot, notebook_handle=True)
svd = TruncatedSVD(n_components=100)

words_2d = svd.fit_transform(data_vectorized.T)

 

df = pd.DataFrame(columns=['x', 'y', 'word'])

df['x'], df['y'], df['word'] = words_2d[:,0], words_2d[:,1], vectorizer.get_feature_names()

 

source = ColumnDataSource(ColumnDataSource.from_df(df))

labels = LabelSet(x="x", y="y", text="word", y_offset=8,

                  text_font_size="8pt", text_color="#555555",

                  source=source, text_align='center')

 

plot = figure(plot_width=600, plot_height=600)

plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)

plot.add_layout(labels)

show(plot, notebook_handle=True)
# Convert to list

df = data.TITLE.values.tolist()



df = [re.sub('\S*@\S*\s?', '', sent) for sent in df]



# Remove new line characters

df = [re.sub('\s+', ' ', sent) for sent in df]



# Remove distracting single quotes

df = [re.sub("\'", "", sent) for sent in df]



pprint(df[:1])
df = [re.sub("-", " ", sent) for sent in df]

df = [re.sub(":", "", sent) for sent in df]
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))



df_words = list(sent_to_words(df))
# Build the bigram and trigram models



bigram = gensim.models.Phrases(df_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[df_words], threshold=100)  



# Faster way to get a sentence clubbed as a trigram/bigram



bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)
def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]



def make_trigrams(texts):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
# Remove Stop Words



data_words_nostops = remove_stopwords(df_words)



# Form Bigrams



data_words_bigrams = make_bigrams(data_words_nostops)



nlp = spacy.load('en', disable=['parser', 'ner'])



data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[11:12]]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=5, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)
pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
# Compute Perplexity



print('\nPerplexity: ', lda_model.log_perplexity(corpus)) 



# Compute Coherence Score



coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
# Visualize the topics



#pyLDAvis.enable_notebook()

#vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

#vis
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=1, limit=6, step=1)
limit=6; start=1; step=1;

x = range(start, limit, step)

plt.figure(figsize=(12,12))

plt.plot(x, coherence_values)

plt.xlabel("Num Topics")

plt.ylabel("Coherence score")

plt.legend(("coherence_values"), loc='best')

plt.show()
for m, cv in zip(x, coherence_values):

    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
# Select the model and print the topics



optimal_model = model_list[3]

model_topics = optimal_model.show_topics(formatted=False)

pprint(optimal_model.print_topics(num_words=10))
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):

   

    sent_topics_df = pd.DataFrame()



   

    for i, row in enumerate(ldamodel[corpus]):

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # -- dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)

    



df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=df)





df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']





df_dominant_topic.head(5)
from gensim.models import CoherenceModel, HdpModel
hdpmodel = HdpModel(corpus=corpus, id2word=id2word)
hdptopics = hdpmodel.show_topics(formatted=False)
hdptopics[0]
len(hdptopics)
from nltk.corpus import stopwords;

import nltk;

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;

from sklearn.decomposition import NMF;

from sklearn.preprocessing import normalize;



vectorizer = CountVectorizer(analyzer='word', max_features=5000, stop_words='english', lowercase=True, 

                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}');

x_counts = vectorizer.fit_transform(title);

print( "Created %d X %d document-term matrix" % (x_counts.shape[0], x_counts.shape[1]) )

transformer = TfidfTransformer(smooth_idf=False);

x_tfidf = transformer.fit_transform(x_counts);
terms = vectorizer.get_feature_names()

print("Vocabulary has %d distinct terms" % len(terms))
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

model = NMF(n_components=5, init='nndsvd');

model.fit(xtfidf_norm)
def get_nmf_topics(model, n_top_words):

    

    feat_names = vectorizer.get_feature_names()

    

    word_dict = {};

    for i in range(num_topics):

        

        words_ids = model.components_[i].argsort()[:-20 - 1:-1]

        words = [feat_names[key] for key in words_ids]

        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;

    

    return pd.DataFrame(word_dict);
num_topics = 5

nmf_df = get_nmf_topics(model, 5)

nmf_df
raw_documents = title.str.strip()

raw_documents= raw_documents.str.lower()

raw_documents = raw_documents.tolist()

raw_doc1 = [i.split() for i in raw_documents]
from sklearn.feature_extraction.text import CountVectorizer

# use a custom stopwords list, set the minimum term-document frequency to 20

vectorizer = CountVectorizer(stop_words = stop_words, min_df = 20) #custom_stop_words

A = vectorizer.fit_transform(raw_documents)

print( "Created %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )
terms = vectorizer.get_feature_names()

print("Vocabulary has %d distinct terms" % len(terms))
from sklearn.feature_extraction.text import TfidfVectorizer

# we can pass in the same preprocessing parameters

vectorizer = TfidfVectorizer(stop_words= stop_words, min_df = 20) #custom_stop_words

A = vectorizer.fit_transform(raw_documents)

print( "Created %d X %d TF-IDF-normalized document-term matrix" % (A.shape[0], A.shape[1]) )
import operator

def rank_terms( A, terms ):

    # get the sums over each column

    sums = A.sum(axis=0)

    # map weights to the terms

    weights = {}

    for col, term in enumerate(terms):

        weights[term] = sums[0,col]

    # rank the terms by their weight over all documents

    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)
ranking = rank_terms( A, terms )

for i, pair in enumerate( ranking[0:20] ):

    print( "%02d. %s (%.2f)" % ( i+1, pair[0], pair[1] ) )
k = 10

# create the model

from sklearn import decomposition

model = decomposition.NMF( init="nndsvd", n_components=k ) 

# apply the model and extract the two factor matrices

W = model.fit_transform( A )

H = model.components_
term_index = terms.index('samsung')

# round to 2 decimal places for display purposes

H[:,term_index].round(2)
import numpy as np

def get_descriptor( terms, H, topic_index, top ):

    # reverse sort the values to sort the indices

    top_indices = np.argsort( H[topic_index,:] )[::-1]

    # now get the terms corresponding to the top-ranked indices

    top_terms = []

    for term_index in top_indices[0:top]:

        top_terms.append( terms[term_index] )

    return top_terms
descriptors = []

for topic_index in range(k):

    descriptors.append( get_descriptor( terms, H, topic_index, 10 ) )

    str_descriptor = ", ".join( descriptors[topic_index] )

    print("Topic %02d: %s" % ( topic_index+1, str_descriptor ) )
%matplotlib inline

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

plt.style.use("ggplot")

matplotlib.rcParams.update({"font.size": 14})
def plot_top_term_weights( terms, H, topic_index, top ):

    # get the top terms and their weights

    top_indices = np.argsort( H[topic_index,:] )[::-1]

    top_terms = []

    top_weights = []

    for term_index in top_indices[0:top]:

        top_terms.append( terms[term_index] )

        top_weights.append( H[topic_index,term_index] )

    # note we reverse the ordering for the plot

    top_terms.reverse()

    top_weights.reverse()

    # create the plot

    fig = plt.figure(figsize=(13,8))

    # add the horizontal bar chart

    ypos = np.arange(top)

    ax = plt.barh(ypos, top_weights, align="center", color="green",tick_label=top_terms)

    plt.xlabel("Term Weight",fontsize=14)

    plt.tight_layout()

    plt.show()
plot_top_term_weights( terms, H, 1, 15 )
def get_top_snippets( all_snippets, W, topic_index, top ):

    # reverse sort the values to sort the indices

    top_indices = np.argsort( W[:,topic_index] )[::-1]

    # now get the snippets corresponding to the top-ranked indices

    top_snippets = []

    for doc_index in top_indices[0:top]:

        top_snippets.append( all_snippets[doc_index] )

    return top_snippets
kmin, kmax = 2, 8
from sklearn import decomposition

topic_models = []

# try each value of k

for k in range(kmin,kmax+1):

    print("Applying NMF for k=%d ..." % k )

    # run NMF

    model = decomposition.NMF( init="nndsvd", n_components=k ) 

    W = model.fit_transform( A )

    H = model.components_    

    # store for later

    topic_models.append( (k,W,H) )
import re

class TokenGenerator:

    def __init__( self, documents, stopwords ):

        self.documents = documents

        self.stopwords = stopwords

        self.tokenizer = re.compile( r"(?u)\b\w\w+\b" )



    def __iter__( self ):

        print("Building Word2Vec model ...")

        for doc in self.documents:

            tokens = []

            for tok in self.tokenizer.findall( doc ):

                if tok in self.stopwords:

                    tokens.append( "<stopword>" )

                elif len(tok) >= 2:

                    tokens.append( tok )

            yield tokens
import gensim

docgen = TokenGenerator(raw_documents, stop_words )

w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=20, sg=1)
def calculate_coherence( w2v_model, term_rankings ):

    overall_coherence = 0.0

    for topic_index in range(len(term_rankings)):

        # check each pair of terms

        pair_scores = []

        for pair in combinations( term_rankings[topic_index], 2 ):

            pair_scores.append( w2v_model.similarity(pair[0], pair[1]))

        # get the mean for all pairs in this topic

        topic_score = sum(pair_scores) / len(pair_scores)

        overall_coherence += topic_score

    # get the mean score across all topics

    return overall_coherence / len(term_rankings)
import numpy as np

def get_descriptor( all_terms, H, topic_index, top ):

    # reverse sort the values to sort the indices

    top_indices = np.argsort( H[topic_index,:] )[::-1]

    # now get the terms corresponding to the top-ranked indices

    top_terms = []

    for term_index in top_indices[0:top]:

        top_terms.append( all_terms[term_index] )

    return top_terms
from itertools import combinations

k_values = []

coherences = []

for (k,W,H) in topic_models:

    # Get all of the topic descriptors - the term_rankings, based on top 10 terms

    term_rankings = []

    for topic_index in range(k):

        term_rankings.append( get_descriptor( terms, H, topic_index, 10 ) )

    # Now calculate the coherence based on our Word2vec model

    k_values.append( k )

    coherences.append( calculate_coherence( w2v_model, term_rankings ) )

    print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ) )
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

plt.style.use("ggplot")

matplotlib.rcParams.update({"font.size": 14})
fig = plt.figure(figsize=(13,7))

# create the line plot

ax = plt.plot( k_values, coherences )

plt.xticks(k_values)

plt.xlabel("Number of Topics")

plt.ylabel("Mean Coherence")

# add the points

plt.scatter( k_values, coherences, s=120)

# find and annotate the maximum point on the plot

ymax = max(coherences)

xpos = coherences.index(ymax)

best_k = k_values[xpos]

plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)

# show the plot

plt.show()
k = best_k

# get the model that we generated earlier.

W = topic_models[k-kmin][1]

H = topic_models[k-kmin][2]
for topic_index in range(k):

    descriptor = get_descriptor( terms, H, topic_index, 10 )

    str_descriptor = ", ".join( descriptor )

    print("Topic %02d: %s" % ( topic_index+1, str_descriptor ) )
lsimodel = LsiModel(corpus=corpus, num_topics=5, id2word=id2word)

lsitopics = lsimodel.show_topics(formatted=False)
ldatopics = lda_model.show_topics(formatted=False)
lsitopics = [[word for word, prob in topic] for topicid, topic in lsitopics]



hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]



ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]
lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=data_lemmatized, dictionary=id2word, window_size=10).get_coherence()



hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=data_lemmatized, dictionary=id2word, window_size=10).get_coherence()



lda_coherence = CoherenceModel(topics=ldatopics, texts=data_lemmatized, dictionary=id2word, window_size=10).get_coherence()
def evaluate_bar_graph(coherences, indices):

    assert len(coherences) == len(indices)

    n = len(coherences)

    x = np.arange(n)

    plt.figure(figsize=(12,12))

    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')

    plt.xlabel('Models')

    plt.ylabel('Coherence Value')
evaluate_bar_graph([lsi_coherence, hdp_coherence, lda_coherence],

                   ['LSI', 'HDP', 'LDA'])