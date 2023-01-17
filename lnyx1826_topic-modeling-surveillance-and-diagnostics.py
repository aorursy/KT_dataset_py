# !pip install scispacy
!pip install guidedlda
# !pip install langdetect
import covid19_tools as cv19 # library generous released to public by Andy White (https://www.kaggle.com/ajrwhite/covid19-tools)
import pandas as pd
import re
from IPython.core.display import display, HTML
import html
import numpy as np
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# import scispacy
import spacy
# import en_core_sci_lg

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.spatial.distance import jensenshannon

import joblib

from IPython.display import HTML, display

from ipywidgets import interact, Layout, HBox, VBox, Box
import ipywidgets as widgets
from IPython.display import clear_output

from tqdm import tqdm
from os.path import isfile

import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use("dark_background")
METADATA_FILE = '../input/CORD-19-research-challenge/metadata.csv'

# Load metadata
meta = cv19.load_metadata(METADATA_FILE)
# print(meta.shape)
# Add tags
meta, covid19_counts = cv19.add_tag_covid19(meta)

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
meta.info()
SOC_ETHIC_TERMS = ['exposure',
                   'immediate',
                   'policy recommendations',                  
                   'mitigation',                   
                   'denominators',                   
                   'testing',                   
                   'sharing information',                   
                   'demographics',
                   'asymptomatic disease',
                   'serosurveys',
                   'convalescent samples',
                   'early detection',
                   'screening',
                   'neutralizing antibodies',
                   'ELISAs',
                   'increase capacity',
                   'existing diagnostic',
                   'diagnostic platforms',
                   'existing surveillance',
                   'surveillance platforms',
                   'recruitment',
                   'support',
                   'coordination',
                   'local expertise',
                   'capacity',
                   'public',
                   'private',
                   'commercial',
                   'non-profit',
                   'academic',
                   'legal',
                   'ethical',
                   'communications',
                   'operational issues',
                   'national guidance',
                   'national guidelines',
                   'universities',
                   'communications',
                   'public health officials',
                   'public',
                   'point-of-care test',
                   'rapid influenza test',
                   'rapid bed-side tests',
                   'tradeoffs',
                   'surveillance experiments',
                   'PCR',
                   'special entity',
                   'longitudinal samples',
                   'ad hoc local interventions',
                   'separation of assay development',
                   'migrate assays',
                   'evolution of the virus',
                   'genetic drift',
                   'mutations',
                   'latency issues',
                   'viral load',
                   'detect the pathogen',
                   'biological sampling',
                   'environmental sampling',
                   'host response markers',
                   'cytokines',
                   'detect early disease',
                   'predict severe disease progression',
                   'best clinical practice',
                   'efficacy of therapeutic interventions',
                   'screening and testing',
                   'policies and protocols',
                   'supplies',
                   'mass testing',
                   'swabs',
                   'reagents',
                   'technology roadmap',
                   'barriers to developing',
                   'market forces',
                   'future coalition',
                   'accelerator models',
                   'Coalition for Epidemic Preparedness Innovations',
                   'streamlined regulatory environment',
                   'CRISPR',
                   'holistic approaches',
                   'genomics',
                   'large scale',
                   'rapid sequencing',
                   'bioinformatics',
                   'genome',
                   'unknown pathogens',
                    'naturally-occurring pathogens',
                   'One Health',
                   'future spillover',
                   'hosts',
                   'ongoing exposure',
                   'transmission hosts',
                   'heavily trafficked',
                   'farmed wildlife',
                   'domestic food',
                   'companion species',
                   'environmental',
                   'demographic',
                   'occupational risk factors',
                   'transmiss', 
                   'transmitted',
                    'incubation',
                    'environmental stability',
                    'airborne',
                    'via contact',
                    'human to human',
                    'through droplets',
                    'through secretions',
                    r'\broute',
                    'exportation'
                  ]

meta, soc_ethic_counts = cv19.count_and_tag(meta,
                                               SOC_ETHIC_TERMS,
                                               'soc_ethic')
print('Loading full text for tag_disease_covid19')
# pulling ~1000 research articles
full_text_repr = cv19.load_full_text(meta[meta.tag_disease_covid19 &
                                          meta.tag_soc_ethic],
                                     '../input/CORD-19-research-challenge')

#pulling ~5000 research articles (picked due to broader search term, which include SARS)
# metadata_filter = meta[meta.tag_soc_ethic == True] 
# full_text_repr = cv19.load_full_text(metadata_filter,
#                                      '../input/CORD-19-research-challenge')
full_text_repr[0]
meta.shape
meta.head()
# meta_rel = meta[meta.tag_disease_covid19 & meta.tag_soc_ethic]
# include only soc and ethic terms
meta_rel = meta[meta.tag_soc_ethic]
meta_rel.shape
# (~(meta_rel['abstract'].isna()))
meta_rel['abstract'].isna().sum()
meta_rel = meta_rel[(~(meta_rel['abstract'].isna()))]
meta_rel.shape
# meta_rel['tag_soc_ethic']== False
metadata_filter = meta[meta.tag_soc_ethic == True] 
#remove non related articles
meta_rel_drop = meta_rel.drop(meta_rel[meta_rel['tag_soc_ethic'] == False].index, inplace=True)
meta_rel.shape
meta_rel['abstract_word_count'] = meta_rel['abstract'].apply(lambda x: len(x.strip().split()))  # word count in abstract
meta_rel['abstract_unique_words']=meta_rel['abstract'].apply(lambda x:len(set(str(x).split())))  # number of unique words in body
meta_rel.head()
# meta_rel['abstract_word_count'] = meta_rel['abstract'].apply(lambda x: len(x.strip().split()))  # word count in abstract
# meta_rel['body_word_count'] = meta_rel['body_text'].apply(lambda x: len(x.strip().split()))  # word count in body
# meta_rel['body_unique_words']= meta_rel['body_text'].apply(lambda x:len(set(str(x).split())))  # number of unique words in body
# meta_rel.head()
# two_terms = ['shelter in place','bed shortage','public health','public interest','human rights','digital rights','face mask','fake news','civil society',
# 'medical treatment','community containment','mental health','suicide hotline','gig worker','medical worker','vulnerable population',
# 'vulnerable community','social distancing','contact tracing','stay at home']
# def replace_space(x):
#     x.replace(" ", "_")
#     print (x)
# replace_space(two_terms)
# from nltk.tree import *

# # Tree manipulation

# # Extract phrases from a parsed (chunked) tree
# # Phrase = tag for the string phrase (sub-tree) to extract
# # Returns: List of deep copies;  Recursive
# def ExtractPhrases( myTree, phrase):
#     myPhrases = []
#     if (myTree.node == phrase):
#         myPhrases.append( myTree.copy(True) )
#     for child in myTree:
#         if (type(child) is Tree):
#             list_of_phrases = ExtractPhrases(child, phrase)
#             if (len(list_of_phrases) > 0):
#                 myPhrases.extend(list_of_phrases)
#     return myPhrases

# test = Tree.parse('(S (NP I) (VP (V enjoyed) (NP my cookies)))')
# print ("Input tree: ", test)

# print ("\nNoun phrases:")
# list_of_noun_phrases = ExtractPhrases(test, 'NP')
# for phrase in list_of_noun_phrases:
#     print (" ", phrase)
# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# abstract_gram = list(sent_to_words(meta_rel['abstract']))

# print(abstract_gram[:1])
# # Build the bigram and trigram models
# bigram = gensim.models.Phrases(abstract_gram, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[abstract_gram], threshold=100)  

# # Faster way to get a sentence clubbed as a trigram/bigram
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# # bigram and trigram example
# # print(bigram_mod[abstract_gram[0]])
# # print(trigram_mod[bigram_mod[abstract_gram[0]]])
# abstract_gram
# print(bigram_mod[abstract_gram[1]])
# # print(trigram_mod[bigram_mod[abstract_gram[0]]])
# include multiple word in the tokenization
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
nltk.download('stopwords')
# en_stop = set(nltk.corpus.stopwords.words('english'))
en_stop = nltk.corpus.stopwords.words('english')
#add new stopwords here
en_stop.extend(['abstract', 'doi', 'preprint', 'copyright','https', 'et', 'al','figure','fig', 'fig.', 
                'al.','PMC', 'CZI','peer', 'reviewed', 'org','author','rights', 'reserved', 'permission', 
                'used', 'using', 'biorxiv', 'medrxiv', 'license','Elsevier','www'])

en_stop = set(en_stop)
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens
meta_rel['tokens'] = meta_rel.apply(lambda x: prepare_text_for_lda(x['abstract']),axis=1)
text_data = list(meta_rel['tokens'])
import seaborn as sns
sns.distplot(meta_rel['abstract_word_count'])
meta_rel['abstract_word_count'].describe()
sns.distplot(meta_rel['abstract_unique_words'])
meta_rel['abstract_unique_words'].describe()
# matrix/ nested list
# text_data
'''Topic model included non SOE term'''
# meta_rel['tag_soc_ethic']== False
len(text_data[1])
from gensim import corpora
#data dictionary
dictionary = corpora.Dictionary(text_data)
#corpus and tf-idf
corpus = [dictionary.doc2bow(text) for text in text_data]


import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
# #readable format of corpus (tf-idf)
# [[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]
'''
passes (int, optional) – Number of passes through the corpus during training.
iterations (int, optional) – Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
Optimize Topic: 8
'''
#10 topics
import gensim
NUM_TOPICS = 8
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, 
#                                            num_topics = NUM_TOPICS, 
#                                            id2word=dictionary, 
#                                            update_every=1,
#                                            passes=25, 
#                                            random_state=7,
#                                            alpha='auto') # TO-Do: Find optimal # topics, iterations, and passes

# ldamodel = gensim.models.ldamodel.LdaModel(corpus, 
#                                            num_topics = NUM_TOPICS, 
#                                            id2word=dictionary, 
#                                            update_every=1,
#                                            passes=100, 
#                                            iterations =100,
#                                            random_state=7,
#                                            alpha='auto') # TO-Do: Find optimal # topics, iterations, and passes

ldamodel = gensim.models.ldamodel.LdaModel(corpus, 
                                           num_topics = NUM_TOPICS, 
                                           id2word=dictionary, 
                                           update_every=1,
                                           passes=25, 
                                           iterations =50,
                                           random_state=7,
                                           alpha='auto') # TO-Do: Find optimal # topics, iterations, and passes


ldamodel.save('model10.gensim')

topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus, 
                                           num_topics = NUM_TOPICS, 
                                           id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=text_data, start=2, limit=40, step=6)
# Show graph
'''    
Pick model (num topics) with the highest coherence score before flattening out.
'''
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
# Select the model and print the topics
optimal_model = model_list[2]
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=10))
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
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=text_data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet
# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model10.gensim')

# lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
# lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
# pyLDAvis.display(lda_display10)
'''Match research paper to one of the 7 LDA topics'''
# #print topic cluster of research articles.
# for i in ldamodel.get_document_topics(corpus)[:]:
#     li = []
#     for j in i:
#         li.append(j[1])
#         bz=li.index(max(li))
#     print(i[bz][0])
#Model perplexity and topic coherence measure topic accuracy. 
# Compute Perplexity # a measure of how good the model is. A low score is good.
print('\nPerplexity: ', ldamodel.log_perplexity(corpus))  

# Compute Coherence Score (Want a hight value)
coherence_model_lda = CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
# import pyLDAvis.gensim
#visual graph
# lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
# # lda matching by words vs matching by documents. Document may contains multiple topics and words. 
# pyLDAvis.display(lda_display)
import pyLDAvis.gensim
lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
#visual graph
lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
# lda matching by words vs matching by documents. Document may contains multiple topics and words. 
pyLDAvis.display(lda_display10)
# for item in full_text_repr[0]['body_text']:
#     print(item['text'])
# text_data_full = []
# for i, record in enumerate(full_text_repr):
#     record_text = "".join([item['text'] for item in record['body_text']])
#     tokens = prepare_text_for_lda(record_text)
#     text_data_full.append(tokens)
# text_data_full
# len(text_data_full[3])
# from gensim import corpora
# dictionary_full = corpora.Dictionary(prepare_text_for_lda)
# corpus_full = [dictionary.doc2bow(text) for text in text_data_full]
# import pickle
# pickle.dump(corpus_full, open('corpus_fulltexts.pkl', 'wb'))
# dictionary_full.save('dictionary_fulltexts.gensim')
# # 5 topics
# import gensim
# NUM_TOPICS = 5
# ldamodel_full = gensim.models.ldamodel.LdaModel(corpus_full, num_topics = NUM_TOPICS, id2word=dictionary_full, passes=15)
# ldamodel_full.save('model10_fulltexts.gensim')
# topics = ldamodel.print_topics(num_words=25)
# for topic in topics:
#     print(topic)
# #10 topics
# import gensim
# NUM_TOPICS = 10
# ldamodel_full = gensim.models.ldamodel.LdaModel(corpus_full, num_topics = NUM_TOPICS, id2word=dictionary_full, passes=15)
# ldamodel_full.save('model10_fulltexts.gensim')
# topics = ldamodel_full.print_topics(num_words=10)
# for topic in topics:
#     print(topic)
# dictionary_full = gensim.corpora.Dictionary.load('dictionary_fulltexts.gensim')
# corpus_full = pickle.load(open('corpus_fulltexts.pkl', 'rb'))
# lda_full = gensim.models.ldamodel.LdaModel.load('model10_fulltexts.gensim')
# import pyLDAvis.gensim
# #visual
# lda_display_full = pyLDAvis.gensim.prepare(lda_full, corpus_full, dictionary_full, sort_topics=False)
# #visual
# pyLDAvis.display(lda_display_full)
import guidedlda 
# seed_topic_list =[
# ['disinformation','misinformation','news','tweet','media','censorship','war','viral','anti-asian','fake'],
# ['police','law','enforcement','liberty','self-determination','force','politics','restriction','freedom','detention','lockdown'],
# ['well-being','isolation','psychological','mental','health','vulnerable','elderly','wellness','trauma','suicide','hotline'],
# ['privacy','surveillance','digital','human','rights','declaration','censorship','self-determination','democracy','discrimination','civil','society'],
# ['economics','economy','gig','low-income','worker','curve','recession','business','jobs','loss'],
# ['healthcare','nurse','doctor','front-line','seniors','caregiver','medical'],
# ['policy','shelter-in-place','GDPR','distancing','contain','containment','suppress','suppression','quarantine','closure','replication','reprecussion','capacity','lockdown']    
# ]     

# seed_topic_list =[     
# ["severe", "symptom", "clinical", "disease", "study", "result", "case", "cov-2", "coronavirus", "covid-19", "cov-2"],
# ["study", "viral", "control", "method", "intervention"],
# ["intervention", "social", "method" ],
# ["china", "wuhan", "country", "hubei", "province", "health", "cov-2", "coronavirus", "covid-19", "cov-2"],
# ["Patient", "patient", "transmission", "epidemic", "measure", "respiratory"],
# ["emergency", "Patient", "patient", "cov-2", "coronavirus", "covid-19", "cov-2", "public", "medical", "outbreak", "case", "number", "estimate", "infection", "health", "disease", "virus", " treatment"],
# ["group", "china", "wuhan", "country", "hubei", "quarantine", "using", "disease"],
# ]
# '''Make our own dataset and word2id'''

# # print(X.shape)
# # print(corpus[100])
# word2id = {}
# vocab = []
# index = 0
# for tx in text_data:
#     for word in tx:
#         if word not in word2id:
#             vocab.append(word)
#             word2id[word] = index
#             index += 1

# print(len(word2id))

# ## transfer corpus to word_ids sentences
# corpus_with_id = []
# max_len = max([len(x) for x in corpus])
# for line in corpus:
#     doc = []
#     for word, fre in line:
#         doc.append(word)
#     doc += [0 for _ in range(max_len - len(doc))]
#     corpus_with_id.append(doc)

# import numpy
# corpus_with_id = numpy.array(corpus_with_id)
# print(corpus_with_id.shape)

# # seed_topics = {}
# # for t_id, st in enumerate(seed_topic_list):
# #     for word in st:
# #         print(word)
# '''Check seed topics for seed_topic_list'''
# seed_topics = {}
# for t_id, st in enumerate(seed_topic_list):
#     for word in st:
#         if word in word2id:
#             seed_topics[word2id[word]] = t_id

# '''model training'''
# model = guidedlda.GuidedLDA(n_topics=7, n_iter=100, random_state=7, refresh=20)
# model.fit(corpus_with_id, seed_topics=seed_topics, seed_confidence=0.15)
# '''Get guidedLDA output'''
# n_top_words = 10
# topic_word = model.topic_word_
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
# '''Retreive the document-topic distributions'''
# doc_topic = model.transform(corpus_with_id)
# for i in range(7):
#     print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
#                                                   ', '.join(np.array(vocab)[list(reversed(corpus_with_id[i,:].argsort()))[0:5]])))

# with open('guidedlda_model.pickle', 'wb') as file_handle:
#      pickle.dump(model, file_handle)
# # load the model for prediction
# # with open('guidedlda_model.pickle', 'rb') as file_handle:
# #      model = pickle.load(file_handle)
# X = guidedlda.datasets.load_data(guidedlda.datasets.NYT) # need to update to main list
# vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
# word2id = dict((v, idx) for idx, v in enumerate(vocab))
# model = guidedlda.GuidedLDA(n_topics=7, n_iter=100, random_state=7, refresh=20)
# seed_topics = {}
# for t_id, st in enumerate(seed_topic_list):
#     for word in st:
#         seed_topics[word2id[word]] = t_id
# model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)
# n_top_words = 10
# topic_word = model.topic_word_
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
# doc_topic = model.transform(X)
# for i in range(9):
#     print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
#                                                   ', '.join(np.array(vocab)[list(reversed(X[i,:].argsort()))[0:5]])))

# model.purge_extra_matrices()
# from six.moves import cPickle as pickle
# with open('guidedlda_model.pickle', 'wb') as file_handle:
#      pickle.dump(model, file_handle)
# # load the model for prediction
# with open('guidedlda_model.pickle', 'rb') as file_handle:
#      model = pickle.load(file_handle)
# doc_topic = model.transform(X)