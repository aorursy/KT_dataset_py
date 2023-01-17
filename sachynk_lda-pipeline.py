import re

import numpy as np

import pandas as pd

from pprint import pprint



# NLTK Stop words

from nltk.corpus import stopwords



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# spacy for lemmatization

import spacy



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

%matplotlib inline



# Enable logging for gensim - optional

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)



# 1. Wordcloud of Top N words in each topic

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import matplotlib.colors as mcolors



from collections import Counter



from matplotlib.ticker import FuncFormatter



# Get topic weights and dominant topics ------------

from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook
stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'may', 'fig' , 'invention', 'first', 'secoTnd' , 'method', 'use', 'say', 'end', 'abstract', 'wherein', 'least'])
df = pd.read_excel("../input/complete-specification/Complete Specification.xlsx")

l = [i for i in df.index if len(df['Complete Specification'][i]) < 50]

len(l)

df.drop(df.index[[l]], inplace = True)

df.reset_index(drop=True,inplace=True) 

df
import os

os.system("pip install git+https://github.com/csurfer/rake-nltk")



##import rake

import operator

import pandas as pd

from nltk import RegexpTokenizer

from rake_nltk import Rake



#law_patents = pd.read_excel("../input/law-patent/law patents without summary.xlsx")

#sample_patents = law_patents.Abstract.apply(lambda x: x.replace("Abstract:\n\n",""))



def patent_keywords(sample_patents):

    r = Rake()

    set_of_keywords = []

    for patent in sample_patents:

        r.extract_keywords_from_text(patent)

        patent_keywords = r.get_ranked_phrases()[:100]

        keyword_str = ""

        for word in patent_keywords:

            keyword_str+= word + " " 

        set_of_keywords.append(keyword_str)

    

    

    sample_patents = set_of_keywords

    tokenizer = RegexpTokenizer(r'\w+')

    sample_patents_tokenize = [w.lower() for w in sample_patents]

    s=" "

    sample_patents_keywords_tokenize = [s.join(tokenizer.tokenize(i)) for i in sample_patents_tokenize]

    return sample_patents_keywords_tokenize
data = patent_keywords(df["Complete Specification"])
# Convert to list

#data = df['Abstract'].values.tolist()



# Remove new line characters

data = [re.sub('\s+', ' ', sent) for sent in data]



# Remove distracting single quotes

data = [re.sub("\'", "", sent) for sent in data]



#pprint(data[:2])
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))



#print(data_words[:2])
# Build the bigram and trigram models

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  



# Faster way to get a sentence clubbed as a trigram/bigram

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)



# See trigram example

#print(trigram_mod[bigram_mod[data_words[1]]])
# Define functions for stopwords, bigrams, trigrams and lemmatization

def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]



def make_trigrams(texts):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
# Remove Stop Words

data_words_nostops = remove_stopwords(data_words)



# Form Bigrams

data_words_bigrams = make_bigrams(data_words_nostops)



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

# python3 -m spacy download en

nlp = spacy.load('en', disable=['parser', 'ner'])



# Do lemmatization keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



#print(data_lemmatized[:2])
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

#print(corpus[:2])
# Human readable format of corpus (term-frequency)

#[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:2]]

#texts[:2]
# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=10, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=50,

                                           alpha='auto',

                                           per_word_topics=True)
# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
lda_model.save('lda_for_patent.model')
# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
def compute_coherence_values(dictionary, corpus, texts, limit, start=5, step=5):

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

        model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,random_state=100,update_every=1,chunksize=100,\

                                                passes=50, alpha='auto', per_word_topics=True, num_topics=num_topics)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
# Can take a long time to run.

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=5, limit=35, step=5)
# Show graph

limit=35; start=5; step=5;

x = range(start, limit, step)

plt.plot(x, coherence_values)

plt.xlabel("Num Topics")

plt.ylabel("Coherence score")

plt.legend(("coherence_values"), loc='best')

plt.show()
# Print the coherence scores

for m, cv in zip(x, coherence_values):

    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
optimal_model = model_list[1]

model_topics = optimal_model.show_topics(formatted=False)

lda_model = optimal_model

pprint(optimal_model.print_topics(num_words=10))
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row_list in enumerate(ldamodel[corpus]):

        row = row_list[0] if ldamodel.per_word_topics else row_list            

        # print(row)

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

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





df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

df_dominant_topic.to_excel('Dominant topic and its percentage contribution in each document.xlsx')

df_dominant_topic.head(10)

# Display setting to show more characters in column

pd.options.display.max_colwidth = 100



sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')



for i, grp in sent_topics_outdf_grpd:

    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 

                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 

                                            axis=0)



# Reset Index    

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)



# Format

sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]



sent_topics_sorteddf_mallet.to_excel('The most representative sentence for each topic.xlsx')

# Show

sent_topics_sorteddf_mallet.head(10)

# Visualize the topics

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

vis
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



cloud = WordCloud(stopwords=stop_words,

                  background_color='white',

                  width=2500,

                  height=1800,

                  max_words=10,

                  colormap='tab10',

                  color_func=lambda *args, **kwargs: cols[i],

                  prefer_horizontal=1.0)



topics = lda_model.show_topics(formatted=False)



fig, axes = plt.subplots(5, 2, figsize=(12,12), sharex=True, sharey=True)



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
topics = lda_model.show_topics(formatted=False)

data_flat = [w for w_list in texts for w in w_list]

counter = Counter(data_flat)



out = []

for i, topic in topics:

    for word, weight in topic:

        out.append([word, i , weight, counter[word]])



df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        



# Plot Word Count and Weights of Topic Keywords

fig, axes = plt.subplots(5, 2, figsize=(10,10), sharey=True, dpi=160)

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

for i, ax in enumerate(axes.flatten()):

    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')

    ax_twin = ax.twinx()

    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')

    ax.set_ylabel('Word Count', color=cols[i])

    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)

    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)

    ax.tick_params(axis='y', left=False)

    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')

    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')



fig.tight_layout(w_pad=2)    

fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    

plt.show()
# Sentence Coloring of N Sentences

def topics_per_document(model, corpus, start=0, end=1):

    corpus_sel = corpus[start:end]

    dominant_topics = []

    topic_percentages = []

    for i, corp in enumerate(corpus_sel):

        topic_percs, wordid_topics, wordid_phivalues = model[corp]

        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]

        dominant_topics.append((i, dominant_topic))

        topic_percentages.append(topic_percs)

    return(dominant_topics, topic_percentages)



dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            



# Distribution of Dominant Topics in Each Document

df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])

dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()

df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()



# Total Topic Distribution by actual weight

topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])

df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()



# Top 3 Keywords for each Topic

topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 

                                 for j, (topic, wt) in enumerate(topics) if j < 3]



df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])

df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)

df_top3words.reset_index(level=0,inplace=True)
# Plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 4), dpi=120, sharey=True)



# Topic Distribution by Dominant Topics

ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')

ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))

tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])

ax1.xaxis.set_major_formatter(tick_formatter)

ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))

ax1.set_ylabel('Number of Documents')

ax1.set_ylim(0, 1000)



# Topic Distribution by Topic Weights

ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')

ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))

ax2.xaxis.set_major_formatter(tick_formatter)

ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))



plt.show()
# Get topic weights

topic_weights = []

for i, row_list in enumerate(lda_model[corpus]):

    topic_weights.append([w for i, w in row_list[0]])



# Array of topic weights    

arr = pd.DataFrame(topic_weights).fillna(0).values



# Keep the well separated points (optional)

arr = arr[np.amax(arr, axis=1) > 0.35]



# Dominant topic number in each doc

topic_num = np.argmax(arr, axis=1)



# tSNE Dimension Reduction

tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

tsne_lda = tsne_model.fit_transform(arr)



# Plot the Topic Clusters using Bokeh

output_notebook()

n_topics = 4

mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 

              plot_width=900, plot_height=700)

plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])

show(plot)