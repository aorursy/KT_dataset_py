%%capture
!pip install kmeanstf
!pip install tabulate
!pip install elasticsearch
import numpy as np
import pandas as pd 
import json
import os
import pickle
import time
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 
import sklearn.metrics as metrics
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import tensorflow as tf
from kmeanstf import KMeansTF
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
import pylab
import matplotlib.cm as cm
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from scipy import spatial
def read_json_data(data_list, input_path): 
    '''
    Inputs: 
        - data_list: json file paths
        - input_path: input_path
        
    Output:
        - dataframe containing: 
              'paper_id', 
              'titles', 
              'abstracts', 
              'introductions', 
              'conclusions', 
              'full_bodytext', 
              'bodysections',
              'body_text_citations', 
              'context_title_list', 
              'cite_start', 
              'cite_end', 
              'cite_mark'
    This function is used to parse json files to return the output elements
    '''
    
    bibentries_title = []
    bibentries_token = []
    for json_file in range(0, len(data_list)):
        bibentries_token.append(list(data_list[json_file]['bib_entries'].keys()))

    for token_num, token_list in enumerate(bibentries_token):
        bibentry_title = []
        for token_len, token in enumerate(token_list):
            bibentry_title.append(data_list[token_num]['bib_entries'][token]['title'])
        bibentries_title.append(bibentry_title)
        
    titles = []
    all_info = []
    paper_id = []
    search_abstracts = []
    for json_file in range(0, len(data_list)):
        paper_id.append(data_list[json_file]['paper_id'])
        titles.append(data_list[json_file]['metadata']['title'])
        all_info.append(data_list[json_file]['body_text'])
        try:
            search_abstracts.append(data_list[json_file]['abstract'])
        except IndexError:
            search_abstracts.append(None)
        except KeyError:
            search_abstracts.append(None)

    abstracts = []
    for texts in search_abstracts:
        local_abstract = []
        if texts is not None:
            for num in range(0, len(texts)):
                local_abstract.append(texts[num]['text'])
        abstracts.append(' '.join(local_abstract))

    bodysections = []
    full_bodytext = []
    introductions = []
    conclusions = []
    cite_tokens = []
    cite_start = []
    cite_end = []
    cite_mark = []
    
    for text_info in all_info:
        local_info = []
        local_cite_token = []
        local_cite_start = []
        local_cite_end = []
        local_cite_mark = []
        local_introduction = []
        local_conclusion = []

        for info_len in range(0, len(text_info)):
            if text_info[info_len]['section'] == 'Introduction':
                local_introduction.append(text_info[info_len]['text'])
            elif text_info[info_len]['section'] == 'Conclusion':
                local_conclusion.append(text_info[info_len]['text'])
            local_info.append(text_info[info_len]['text'])
        for indices in text_info:
            for cite_spans in indices['cite_spans']:
                local_cite_token.append(cite_spans['ref_id'])
                local_cite_start.append(cite_spans['start'])
                local_cite_end.append(cite_spans['end'])
                try:
                    local_cite_mark.append(cite_spans['text'])
                except KeyError:
                    local_cite_mark.append(None)
        introductions.append(''.join(local_introduction))
        conclusions.append(''.join(local_conclusion))
        full_bodytext.append(' '.join(local_info))
        bodysections.append(local_info)
        cite_tokens.append(local_cite_token)
        cite_start.append(local_cite_start)
        cite_end.append(local_cite_end)
        cite_mark.append(local_cite_mark)

    bib_dict_list = []
    for bib_ref, bib_ttl in (zip(bibentries_token, bibentries_title)):
        bib_dict = {}
        for bib_bib_ref, bib_bib_ttl in zip(bib_ref, bib_ttl):
            bib_dict[bib_bib_ref] = bib_bib_ttl
        bib_dict_list.append(bib_dict)

    context_title_list = []
    for cite_val, bib_val in (zip(cite_tokens, bib_dict_list)):
        cite_set = cite_val
        bib_set = set(bib_val)
        context_title_temp = []
        for value in cite_set:
            for val in bib_set:
                if value == val:
                    context_title_temp.append(bib_val[value])
                elif value == None:
                    context_title_temp.append(None)
                    break
        context_title_list.append(context_title_temp)
        
    
    fields = {
              'paper_id': paper_id[0], 
              'titles': titles[0], 
              'abstracts': abstracts[0], 
              'introductions': introductions[0], 
              'conclusions': conclusions[0], 
              'full_bodytext': full_bodytext[0], 
              'bodysections': bodysections[0],
              'context_title_list': context_title_list[0], 
              'cite_start': cite_start[0], 
              'cite_end': cite_end[0], 
              'cite_mark': cite_mark[0]
            }
    return fields
input_path = '/kaggle/input/CORD-19-research-challenge/'

start = time.time()
f_name_list = ['biorxiv_medrxiv/biorxiv_medrxiv', 'comm_use_subset', 'noncomm_use_subset', 'custom_license/custom_license']
files = []
for f_name in f_name_list:
    for root, dirs, file in os.walk(input_path + f_name):
        for fname in file:
            files.append(os.path.abspath(os.path.join(root, fname)))  

file_list = []
folder_list = []
for l in files:
    with open(l) as json_data_l:
        data_list = json.load(json_data_l)
        context_data = read_json_data([data_list], input_path=input_path)
    file_list.append(context_data)
folder_list.append(file_list)

flatten_data = [item for sublist in folder_list for item in sublist]
context_data = pd.DataFrame.from_dict(flatten_data)

print('Elapsed time: ', time.time() - start)

# pickle.dump(context_data, open('context_data.df', 'wb'))
# print('Data file size: ', os.path.getsize('context_data.df')/(1024*1024))
context_data.head(3)
context_data.tail(3)

def create_features_tfidf(data, columns=['title'], max_features=None):
    corpus = pd.DataFrame(columns=['text'])
    corpus['text'] = data[columns].dropna().agg('\n'.join, axis=1)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)

    x = vectorizer.fit_transform(corpus['text'].values)
    
    return x, vectorizer


columns_to_consider = ['titles', 'abstracts', 'full_bodytext']
x, vectorizer = create_features_tfidf(context_data, columns=columns_to_consider, max_features=100)

print(x.shape)
# pickle.dump(x, open('tfidf_matrix.pkl', 'wb'))
features = tf.convert_to_tensor(x.toarray(), dtype=tf.float32)
km = KMeansTF(n_clusters=10, random_state=1)
km.fit(features)
labels = km.predict(x.toarray())
s_score = metrics.silhouette_score(x.toarray(), labels)
db_score = metrics.davies_bouldin_score(x.toarray(), labels)

print(s_score, db_score)
pickle.dump([km, labels], open('clustering_kernel.pkl', 'wb'))
'''
[km, labels] = pickle.load(open('clustering_kernel.pkl', 'rb'))
x = pickle.load(open('tfidf_matrix.pkl', 'rb'))

colors = cm.rainbow(np.linspace(0, 10, 100))


################### 3D Visualization ######################
# Use t-SNE to reduce dimensions down to 3
tsne = TSNE(n_components=3, verbose=0).fit(x)
tsne_vec_3d = tsne.embedding_

fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
ax_3d.scatter(tsne_vec_3d[:, 0], tsne_vec_3d[:, 1], tsne_vec_3d[:, 2], c=colors[labels-1])#c=kernel.labels_*100, cmap=pylab.cm.cool)
ax_3d.set_title('t-SNE')


# Use PCA to reduce dimensions down to 3
pca = PCA(n_components=3)
pca.fit(x.toarray())
#print('Explained variance ratio of the first three PCs: ', pca.explained_variance_ratio_)
pca_vec_3d = pca.transform(x.toarray())

ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
ax_3d.scatter(pca_vec_3d[:, 0], pca_vec_3d[:, 1], pca_vec_3d[:, 2], c=colors[labels-1])
ax_3d.set_title('PCA')



################### 2D Visualization ######################
# Use t-SNE to reduce dimensions down to 2
tsne = TSNE(n_components=2, verbose=0).fit(x)
tsne_vec_2d = tsne.embedding_

ax_2d=plt.subplot(2, 2, 3)
ax_2d.scatter(tsne_vec_2d[:, 0], tsne_vec_2d[:, 1], c=colors[labels-1])
ax_2d.set_title('t-SNE')


# Use PCA to reduce dimensions down to 2
pca = PCA(n_components=2).fit(x.toarray())
#print('Explained variance ratio of the first two PCs: ', pca.explained_variance_ratio_)
pca_vec_2d = pca.transform(x.toarray())

ax_2d=plt.subplot(2, 2, 4)
ax_2d.scatter(pca_vec_2d[:, 0], pca_vec_2d[:, 1], c=colors[labels-1])
ax_2d.set_title('PCA')
'''

import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import re
from pprint import pprint
documents = ['Efforts targeted at a universal coronavirus vaccine.',
'Exploration of use of best animal models and their predictive value for a human vaccine.',
'Methods evaluating potential complication of Antibody Dependent Enhancement (ADE) in vaccine recipients.',
'Approaches to evaluate risk for enhanced disease after vaccination',
'Effectiveness of drugs being developed and tried to treat COVID-19 patients.',
'Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocycline that may exert effects on viral replication.',
'Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.',
'Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.',
'Efforts to develop animal models and standardize challenge studies',
'Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers',
'Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models (in conjunction with therapeutics)',]

words=[]
for c in documents: 
    words.append(c.split(' '))

words = [row.split() for row in documents]
from gensim.models.phrases import Phrases, Phraser

bigram = Phrases(words, min_count=30, progress_per=10000)
trigram = Phrases(bigram[words], threshold=100)
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

bigrams = [b for l in documents for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
stop_words = stopwords.words('english')
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

words_nostops = remove_stopwords(words)
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
words_bigrams = make_bigrams(words_nostops)
id2word = corpora.Dictionary(words_bigrams)
texts = words_bigrams
corpus = [id2word.doc2bow(text) for text in texts]
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=10,
                                       chunksize=5,
                                       passes=1,
                                       per_word_topics=True)
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
lda_model.print_topics(20,num_words=15)[:10]

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=words_bigrams):
    sent_topics_df = pd.DataFrame()
    keywords_arr = []
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                keywords_arr.append(topic_keywords)
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df, keywords_arr)
df_topic_sents_keywords, keywords_arr = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=words_bigrams)
df_topic_sents_keywords

keywords1 = [i.lower().replace(',', '').split() for i in keywords_arr]
keywords2= [x for sublist in keywords1 for x in sublist]
keywords = keywords2
from tqdm import tqdm
import glob
import json
import re

df = pd.DataFrame()
df['body'] = context_data['full_bodytext']
df['paper_id'] = context_data['paper_id']
df['abstract'] = context_data['abstracts']
df['title'] = context_data['titles']

Dic_title_to_key={}
for i in range(len(df.body)):
    for item in keywords:
        if item.lower() in df.body[i].lower():
            Dic_title_to_key[df.title[i]]=item
df2=pd.DataFrame(Dic_title_to_key.items(), columns=['titles', 'keyword'])
df3=pd.merge(df2, context_data, on='titles')
df3 = df3.sort_values('keyword').reset_index().drop(columns=['index'])
df3.to_csv (r'export_dataframe.csv', index = False, header=True)


# [km, labels] = pickle.load(open('clustering_kernel.pkl', 'rb'))

clustering_data = context_data[columns_to_consider].dropna().reset_index()
clustering_data['label'] = labels

tm_data = pd.read_csv('export_dataframe.csv')
tm_titles = set(tm_data['title'].values)

table = []
for l in np.unique(labels):
    label_data = clustering_data[clustering_data['label'] == l]
    label_titles = set(label_data['title'].values)

    selected_articles_i = tm_titles.intersection(label_titles)
    selected_articles_u = tm_titles.union(label_titles)
    table.append([l, len(label_titles), len(selected_articles_i), len(selected_articles_u)])

print(tabulate(table, headers=["Cluster ID","# of Articles", "Intersection", "Union"]))


l = 3
label_data = clustering_data[clustering_data['label'] == l]
label_titles = set(label_data['title'].values)

selected_articles_i = set(tm_titles.intersection(label_titles))
selected_articles_u = set(tm_titles.union(label_titles))

selected_articles = clustering_data[clustering_data['title'].isin(selected_articles_u)]
selected_articles = selected_articles.filter(items=['title', 'sha'])

# pickle.dump(selected_articles, open('selected_articles.pkl', 'wb'))

from nltk.tokenize import word_tokenize
import string

stop_words = set(stopwords.words('english'))

tokenized_words = []
for words in documents:
    tokenized_words.append(word_tokenize(words))
    
wordsFiltered = []

for l in tokenized_words:
    temp_list = []
    for w in l:
        if w not in stop_words:
            temp_list.append(w)
    wordsFiltered.append(temp_list)
    
search_sentences = []
for i in wordsFiltered:
    search_sentences.append(' '.join(i))

remove_punct = []
for s in search_sentences:
    remove_punct.append(s.translate(str.maketrans('', '', string.punctuation)))
!git clone https://github.com/facebookresearch/InferSent.git
from InferSent.models import InferSent
import torch
nltk.download('punkt')   

!mkdir encoder
!curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = '/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)

infersent.build_vocab_k_words(K=100000)
string_titles = []
for sent in list(context_data['titles']):
    if isinstance(sent, str):
        string_titles.append(sent)

infersent.update_vocab(string_titles)
for topic in remove_punct[:4]:
    infersent.visualize(topic, tokenize=True)
index_mappings = {
      "settings": {
          "number_of_shards": 2,
          "number_of_replicas": 1
        },
       "mappings": {
          "dynamic": "true",
           "_source": {
                 "enabled": "true"
             },
          "properties": {
              "paper_id": {
                  "type": "keyword"
              },  
              "title": {
                  "type": "text"
              },
              "abstract": {
                  "type": "text"
              },
              "title_vector": {
                "type": "dense_vector",
                "dims": 512
              },
              "abstract_vector": {
                "type": "dense_vector",
                "dims": 512
              }
            }
          }
        }
%%capture
import sys
!{sys.executable} -m pip install tensorflow_hub
!{sys.executable} -m pip install tensorflow
# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
# We will be feeding 1D tensors of text into the graph.
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    embeddings = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()
# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)
# Use tensorflow for the Universal Sentence Encoder
GPU_LIMIT = 0.5

print("Downloading pre-trained embeddings from tensorflow hub...")
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2") # You can also download the model and reload it from your path
text_ph = tf.placeholder(tf.string)
embeddings = embed(text_ph)
print("Done.")

print("Creating tensorflow session...")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_LIMIT
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())
print("Done.")
# Uncomment the following to install the elasticsearch python package
import sys
!{sys.executable} -m pip install elasticsearch
articles_list = "selected_articles.pkl"
articles_df = pd.read_pickle(articles_list)
titles = articles_df['title'].tolist()
print(len(titles))
def load_article(context_data, title):
    doc = context_data[context_data['titles'] == title]
    try:
        return doc.iloc[0]
    except:
        return None
INDEX_NAME = "covid19"
INDEX_Mappings = index_mappings

BATCH_SIZE = 1000 # The batch size for bulk indexing

client = Elasticsearch() # Connect to the elasticsearch service we started earlier
client.indices.delete(index=INDEX_NAME, ignore=[404]) # Delete the index if it exists
client.indices.create(index=INDEX_NAME, body=INDEX_Mappings)
# Embedding method
def embed_text(text):
    vectors = session.run(embeddings, feed_dict={text_ph: text})
    return [vector.tolist() for vector in vectors]
# Extract content and apply sentence embedding, then bulk index
def index_batch(docs):
    titles = [doc["titles"] for doc in docs]
    title_vectors = embed_text(titles)
    
    abstracts = [doc["abstracts"] for doc in docs]
    abstract_vectors = embed_text(abstracts)

    requests = []
    for i, doc in enumerate(docs):
        request = {}
        request['paper_id'] = doc['paper_id']
        request['title'] = titles[i] 
        request['abstract'] = abstracts[i]
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = title_vectors[i]
        request["abstract_vector"] = abstract_vectors[i]
        requests.append(request)
    bulk(client, requests)
docs = []
count = 0

start_time = time.time()
print(len(titles))
for title in titles:
    doc = load_article(data, title)
    
    if doc is None:
        continue
    docs.append(doc)
    count += 1

    if count % BATCH_SIZE == 0:
        print(count)
        index_batch(docs)
        docs = []
        print("Indexed {} documents.".format(count))

if docs:
    index_batch(docs)
    print("Indexed {} documents.".format(count))

client.indices.refresh(index=INDEX_NAME)
print("Done indexing.")
end_time = time.time()
print("Time to index: ", (end_time - start_time)/3600)
def run_query_loop(SEARCH_SIZE=5):
    while True:
        try:
            handle_query(SEARCH_SIZE)
        except KeyboardInterrupt:
            return
def get_top_paragraphs(data, titles, query_vector, SEARCH_SIZE):  
    enhanced_origin = {}
    for title in titles:
        doc = load_article(data, title)
        body_sections = doc['bodysections']
        citation_titles = doc['context_title_list']
        cite_mark = doc['cite_mark']
        mark_title = {}
        for i, c_mark in enumerate(cite_mark):
            mark_title[c_mark] = citation_titles[i]
            
        for sec in body_sections:
            enhan = sec
            for m in mark_title:
                try:
                    enhan = enhan.replace(m, mark_title[m])
                except:
                    pass
            enhanced_origin[enhan] = title + "<TITLE_SEP>" + sec

    paragraphs_all = list(enhanced_origin.keys())
    paragraphs_all_vectors = embed_text(paragraphs_all)
    scores = []
    score_index = {}
    for p in paragraphs_all_vectors:
        score = 1 - spatial.distance.cosine(p, query_vector)
        scores.append(score)
    for i, s in enumerate(scores):
        score_index[s] = i
    scores.sort(reverse=True)
    topk_scores = scores[:SEARCH_SIZE]
    topk_index = [score_index[s] for s in topk_scores]
    for i, index in enumerate(topk_index):
        print("Top {} Relevant Paragraph: ".format(i))
        top_enhanced_paragraph = paragraphs_all[index]
        top_origin_paragraph = enhanced_origin[top_enhanced_paragraph]
        title, paragraph = top_origin_paragraph.split('<TITLE_SEP>')[0],top_origin_paragraph.split('<TITLE_SEP>')[1]
        print("title: ", title)
        print("paragraph: ", paragraph)
        print("*" * 50)
def handle_query(SEARCH_SIZE):
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + cosineSimilarity(params.query_vector, doc['abstract_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["paper_id","title", "abstract", "body_texts"]}
        }
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    print()
    print("The top " + str(SEARCH_SIZE) + " relevant abstracts to the question")
    count = 0
    top_titles = []
    for hit in response["hits"]["hits"]:
        count += 1
        title = hit["_source"]["title"]
        print("Top ", str(count))
        print("Paper Id: ", hit["_source"]["paper_id"])
        print("Paper Title: ", title)
        print("Similarity Score to the query: ", hit["_score"])
        print("Abstract Text: ", hit["_source"]["abstract"])
        print("=" * 50)
        top_titles.append(title)
    print()
    print("The top " + str(SEARCH_SIZE) + " paragraphs from the top abstracts articles")
    get_top_paragraphs(data, top_titles, query_vector, SEARCH_SIZE)
run_query_loop()
print("Closing tensorflow session...")
session.close()
