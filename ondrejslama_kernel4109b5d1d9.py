!pip install spacy scispacy spacy_langdetect https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_lg-0.2.3.tar.gz --user
!python -m spacy download 'en_core_web_sm' --user
!python -m spacy link en_core_web_sm en 
!pip install wordcloud --user
!pip install nltk --user
!pip install gensim --user
!pip install bert-extractive-summarizer --user
!pip install spacy>=2.1.3 --user
!pip install transformers>=2.2.2 --user
!pip install neuralcoref --user
!python -m spacy download en_core_web_md --user
import numpy as np 
import pandas as pd

import glob
import json

import scispacy
import spacy
import en_core_sci_lg as en_core_sci_md
from spacy_langdetect import LanguageDetector
import en_core_web_sm

import pickle
import os
import csv
import re
import datetime

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

from scipy.spatial.distance import jensenshannon

import joblib
from IPython.display import HTML, display, Image

from ipywidgets import interact, Layout, HBox, VBox, Box
import ipywidgets as widgets
from IPython.display import clear_output

import string
import nltk
import configparser
from gensim.models import FastText
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
tqdm().pandas()

from summarizer import Summarizer
from wordcloud import WordCloud
Image(url='https://images.jifo.co/22486206_1586441876420.png')
# source: https://www.who.int/emergencies/diseases/novel-coronavirus-2019
## PLEASE UPDATE YOUR DATA PATH HERE
root_path = '/hackathon/covid-19/data'
# load metadata
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head(1)
# load all json
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json) # should be 88626
# read json files
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            if 'abstract' in content:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
    
dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}
for entry in tqdm(all_json):
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
papers = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])
papers.head(1)
# merge with metadata
df = pd.merge(papers, meta_df, left_on='paper_id', right_on='sha', how='left').drop('sha', axis=1)
# use abstracts from the metadata when possible, fill the missing values with the abstract from the extracted values from the JSON file.
df.loc[df.abstract_y.isnull() & (df.abstract_x != ''), 'abstract_y'] = df[(df.abstract_y.isnull()) & (df.abstract_x != '')].abstract_x
df.rename(columns = {'abstract_y': 'abstract'}, inplace=True)
df.drop('abstract_x', axis=1, inplace=True)
# drop duplicates
df.drop_duplicates(['paper_id', 'body_text'], inplace=True)
# language detection of abstracts
nlp = en_core_sci_md.load(disable=["tagger", "ner"])
nlp.max_length = 2000000
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
# keep only english language of abstracts
df['abstract_lang'] = df.abstract.progress_apply(lambda x: nlp(str(x))._.language['language'])
# language only english language of body text
df['body_lang'] = df.body_text.progress_apply(lambda x: nlp(str(x[:2000]))._.language['language'])
# graph by language
by_lang = df['body_lang'].value_counts().sort_values(ascending=False).head(5)
fig, jd = plt.subplots(figsize=(10, 5))
jd.spines["top"].set_visible(False)
jd.spines["right"].set_visible(False)

by_lang.plot(ax=jd, title="Articles by Publication", kind='bar', color='#0066CC')
df = df[(df.abstract_lang == 'en') & (df.body_lang == 'en')]
df.head()
df['publish_year'] = pd.DatetimeIndex(df['publish_time']).year
# medium model
nlp = en_core_sci_md.load(disable=["tagger", "parser", "ner"])
nlp.max_length = 2000000

customize_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.', 
    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si'
]

# Mark them as stop words
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True


def lemmatize_text(text, nlp=nlp):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower() for token in doc 
                         if not (token.like_num or token.is_stop or token.is_punct or token.is_space or len(token)==1)]
    return " ".join(lemmatized_tokens)


def normalize_corpus(corpus, nlp=nlp):
    txt = lemmatize_text(corpus, nlp)
    return txt
df['preprocessed_body_text'] = df['body_text'].progress_apply(normalize_corpus)
df.reset_index(inplace = True)
fl = open("../../data/processed/processed_data.pkl", "wb")
pickle.dump(df, fl)
fl.close()
df.to_csv("../../data/processed/processed_data.csv")
df.head(2)
source_data = df['source_x'].value_counts().sort_values(ascending=False)
fig, sd = plt.subplots(figsize=(7, 5))
sd.spines["top"].set_visible(False)
sd.spines["right"].set_visible(False)

source_data.plot(ax=sd, title="Articles by Source", kind='bar', color='#0066CC')
by_journal = df['journal'].value_counts().sort_values(ascending=True).tail(25)
fig, jd = plt.subplots(figsize=(20, 15))
jd.spines["top"].set_visible(False)
jd.spines["right"].set_visible(False)

by_journal.plot(ax=jd, title="Articles by Publication", kind='barh', color='#0066CC')
df['publish_time'] = pd.to_datetime(df['publish_time'])
df_pb = df[df['publish_time']<=datetime.date(2020,4,30)]
by_pub_time =  df_pb['publish_time'].value_counts().sort_index().groupby(pd.Grouper(freq='M')).sum()[-50:]
by_pub_time.index = by_pub_time.index.date

fig, ax = plt.subplots(figsize=(25, 7))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

by_pub_time.plot(ax=ax, title="Articles by Time of Publication", kind='bar', color='#0066CC')
def tokenize_text(text):
    return [token.strip() for token in nltk.word_tokenize(text)]
nltk.download('punkt')
data_title = df[df['title'].notnull()]
data_title['processed_title'] = data_title['title'].progress_apply(normalize_corpus)
data_title.reset_index(inplace=True)
words = pd.Series(np.concatenate([tokenize_text(x) for x in data_title['processed_title']])).value_counts()
def wordcloud_plot(df):
    np.random.seed(64)
    
    wc = WordCloud(
    background_color="white",
    max_words=150,
    max_font_size=35,
    scale=5,
    random_state=0).generate_from_frequencies(df)
    
    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')
    
    plt.imshow(wc)
    plt.show()
wordcloud_plot(words)
# loading if needed
fl = open("../../data/processed/processed_data.pkl", "rb")
df = pickle.load(fl)
# compute tf-idf
vectorizer = TfidfVectorizer(max_features=2**12)
X = vectorizer.fit_transform(tqdm(df['preprocessed_body_text']))
WCSS = []
for i in tqdm(range(1, 40)):
    km = MiniBatchKMeans(n_clusters=i, random_state=0)
    km.fit(X)
    WCSS.append(km.inertia_)
plt.plot(range(1, 40), WCSS, marker='o')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.show()
# run k-means with k = 18
k = 18
kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
y_pred = kmeans.fit_predict(X)
# add clusters to data frame
df_cluster = pd.DataFrame(y_pred, columns = ["cluster_k_means"])
df = pd.merge(df, df_cluster, left_index = True, right_index = True)
# save it
fl = open("../../data/processed/df_with_cluster.pkl", "wb")
pickle.dump(df, fl)
fl.close()
df.to_csv("../../data/processed/df_with_cluster.csv")
# loading if needed
fl = open("../../data/processed/df_with_cluster.pkl", "rb")
df = pickle.load(fl)
keywords_therapeutic = [
    'clinical trial',
    'therapeutic',
    'therapy',
    'clinical effectiveness',
    'actemra',
    'ADE',
    'inflammatory response',
    'ACE-2',
    'Camostat Mesilate',
    'interferon beta',
    'interferon alpha',
    'tocilizumab']
keywords_vaccine = ['vaccine',
                    'vaccines',
                    'dna',
                    'inactivated',
                    'live attenuated virus',
                    'lav',
                    'non-replicating viral vector',
                    'protein subunit',
                    'replicating viral vector',
                    'rna',
                    'virus-like particle',
                    'vlp',
                    'viral',
                    'anti-viral',
                    'inhibitor',
                    'immune',
                    'antibodies',
                    'adenoviruses',
                    'antigen',
                    'lentiviral',
                    'mrna', 
                    'naproxen', 
                    'clarithromycin',  
                    'minocyclinethat']
keywords_general = ['covid-19', 
                    'coronavirus', 
                    '2019-ncov' , 
                    'covs', 
                    'sars cov2', 
                    'coronae', 
                    'positive-sense rna viruses']
keywords_ade = ['ADE', 
                'Antibody-Dependent Enhancement', 
                'variable S domain', 
                's protein', 
                'p0dtc2', 
                'Spike glycoprotein', 
                'amino acid residue variation analysis', 
                'APC', 
                'Autophagy inhibitors', 
                'HL-CZ human promonocyte cell line', 
                'TNF-α', 
                'IL-4', 
                'IL-6']
keywords_animal = ['animal models', 
                   'mouse', 
                   'guinea pigs', 
                   'golden Syrian hamsters', 
                   'ferrets', 
                   'rabbits', 
                   'rhesus macaques', 
                   'marmosets', 
                   'cats']
keywords_drugs = ['viral inhibitor', 
                  'viral replication', 
                  'clinical trial', 
                  'bench trial', 
                  'clarithromycin', 
                  'minocycline', 
                  'remdesivir',
                  'hydroxychloroquine',
                  'azithromycin', 
                  'lopinavir', 
                  'nsp5', 
                  'C30 Endopeptidase', 
                  'ritonavir', 
                  'actemra', 
                  'naproxen']
keywords = keywords_general + keywords_therapeutic + keywords_vaccine + keywords_ade + keywords_animal + keywords_drugs 
def split_sentences(text):
    sentences_string = []
    for line in text.split('\n'):
        l = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_string.extend(sent for sent in l.groupdict()['postcolon'].split('.') if sent)
    sentences = []
    for sentence in sentences_string:
        token = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).split()
        token = [tok for tok in token if len(tok) > 2]
        token = [x for x in token if not x.isdigit()]
        if len(token) > 2:
            sentences.append(token)
    return sentences
df['sentences'] = df['preprocessed_body_text'].progress_apply(split_sentences) 
def flatten(lists):
    results = []
    for row in lists:
        for inside_list in row:
            results.append(inside_list) 
    return results
input_model = flatten(df['sentences'].to_list()) 
model = FastText(size=1000, window=3, min_count=5, workers=4, min_n=2, hs=1)
model.build_vocab(input_model)
model.train(input_model, total_examples=model.corpus_count, epochs=1) 
model.save("../models/fasttext.model") 
model.wv.most_similar("covid-19", topn=5) 
def generate_keywords(keywords_list):
    model_keys = []
    for key in keywords_list:
        key_generated = model.wv.most_similar(key, topn=5)
        model_keys.append([i[0] for i in key_generated])
    return model_keys
keywords_general_plus = flatten(generate_keywords(keywords_general)) 
keywords_plus = flatten(generate_keywords(keywords)) 
# save it
fl = open("../../data/processed/keywords_general_plus.pkl", "wb")
pickle.dump(keywords_general_plus, fl)
fl.close()

fl = open("../../data/processed/keywords_plus.pkl", "wb")
pickle.dump(keywords_plus, fl)
fl.close()
# loading if needed
fl = open("../../data/processed/df_with_cluster.pkl", "rb")
df = pickle.load(fl)
# Count vectorizer works better than tfidf for lda
vectorizer = CountVectorizer(max_features=2**12)
data_vectorized = vectorizer.fit_transform(tqdm(df['preprocessed_body_text']))
lda = LatentDirichletAllocation(n_components=20, random_state=0, n_jobs=-1, verbose=True)
lda.fit(data_vectorized)
joblib.dump(lda, '../models/lda.csv')
def print_top_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        message = "\nTopic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
print_top_words(lda, vectorizer, n_top_words=10)
doc_topic_dist = pd.DataFrame(lda.transform(data_vectorized))
doc_topic_dist.to_csv('../models/doc_topic_dist.csv', index=False)
# merge to df
doc_topic_dist["cluster_lda"] = doc_topic_dist.apply(lambda x: doc_topic_dist.columns[x.idxmax()], axis = "columns")
df = pd.merge(df, doc_topic_dist[["cluster_lda"]], left_index = True, right_index = True)
# save it
fl = open("../../data/processed/df_with_cluster_both.pkl", "wb")
pickle.dump(df, fl)
fl.close()
df.to_csv("../../data/processed/df_with_cluster_both.csv")
# loading if needed
fl = open("../../data/processed/df_with_cluster_both.pkl", "rb")
df = pickle.load(fl)
# filtering function
def filter_papers_word_list(word_list, data, threshold = 0.3, keywords_general = keywords_general):
    papers_id_list = []
    for idx, paper in tqdm(data.iterrows(), total=df.shape[0]):
        keys_in_text = [x in paper.body_text for x in word_list]
        keys_general_in_text = [x in paper.body_text for x in keywords_general]
        if (sum(keys_in_text) / len(word_list) > threshold) & any(keys_general_in_text):
            papers_id_list.append(paper.paper_id)
    return papers_id_list
df.shape
# get cluster proportions
def cluster_prop(data, var_name):
    prop = data[var_name].value_counts().sort_values(ascending = False).to_frame().rename(columns = {var_name: "count"})
    prop["prop_count"] = prop["count"] / prop["count"].sum()
    return prop
# iteratively change threshold -> rerun
filtered_general = filter_papers_word_list(keywords, df, threshold = 0.375)
df_keywords = df[df['paper_id'].isin(filtered_general)]
df_keywords.shape
cluster_prop(df, "cluster_lda").head(6)
cluster_prop(df_keywords, "cluster_lda").head(6)
cluster_prop(df, "cluster_k_means").head(6)
cluster_prop(df_keywords, "cluster_k_means").head(6)
# iteratively change threshold -> rerun
filtered_general = filter_papers_word_list(keywords_plus, df, threshold = 0.15)
df_keywords = df[df['paper_id'].isin(filtered_general)]
df_keywords.shape 
cluster_prop(df_keywords, "cluster_lda").head(6) 
cluster_prop(df_keywords, "cluster_k_means").head(6) 
papers_8_1 = filter_papers_word_list(keywords, df, threshold = 0.38)
df_keywords_th_04 = df[df['paper_id'].isin(papers_8_1)]
df_keywords_th_04.shape
fl = open("../../data/processed/df_filtered_threshold-04_keywords-all.pkl", "wb")
pickle.dump(df_keywords_th_04, fl)
fl.close()
# need to be re-run

# papers_8_2 = filter_papers_word_list(keywords, df, threshold = 0.3)
# df_keywords_th_03_clust = df[df['paper_id'].isin(papers_8_2) & 
#                              df["cluster_lda"].isin([15, 7, 12, 6]) & 
#                              df["cluster_k_means"].isin([16, 9, 12, 6])]
# df_keywords_th_03_clust.shape 
# fl = open("../../data/processed/df_filtered_threshold-03_both-clusters_keywords-all.pkl", "wb")
# pickle.dump(df_keywords_th_03_clust, fl)
# fl.close()
# need to be re-run

# papers_8_3 = filter_papers_word_list(keywords_plus, df, threshold = 0.15)
# df_keywords_plus_th_015 = df[df['paper_id'].isin(papers_8_3)]
# df_keywords_plus_th_015.shape
# fl = open("../../data/processed/df_filtered_threshold-015_keywords-all-plus.pkl", "wb")
# pickle.dump(df_keywords_plus_th_015, fl)
# fl.close()
doc_topic_dist = pd.read_csv('../models/doc_topic_dist.csv')
def get_k_nearest_docs(doc_dist, k=5, lower=1950, upper=2020, only_covid19=False, get_dist=False):
    '''
    doc_dist: topic distribution (sums to 1) of one article
    
    Returns the index of the k nearest articles (as by Jensen–Shannon divergence in topic space). 
    '''
    
    relevant_time = df.publish_year.between(lower, upper)
    
    if only_covid19:
        temp = doc_topic_dist[relevant_time & is_covid19_article]
        
    else:
        temp = doc_topic_dist[relevant_time]
         
    distances = temp.apply(lambda x: jensenshannon(x, doc_dist), axis=1)
    k_nearest = distances[distances != 0].nsmallest(n=k).index
    
    if get_dist:
        k_distances = distances[distances != 0].nsmallest(n=k)
        return k_nearest, k_distances
    else:
        return k_nearest
def recommendation(paper_id, k=5, lower=1950, upper=2020, only_covid19=False, plot_dna=False):
    '''
    Returns the title of the k papers that are closest (topic-wise) to the paper given by paper_id.
    '''

    recommended, dist = get_k_nearest_docs(doc_topic_dist[df.paper_id == paper_id].loc[0], k, lower, upper, only_covid19, get_dist=True)
    recommended = df.iloc[recommended].copy()
    recommended['similarity'] = 1 - dist 
    
    h = '<br/>'.join(['<a href="' + l + '" target="_blank">'+ n + '</a>' +' (Similarity: ' + "{:.2f}".format(s) + ')' for l, n, s in recommended[['url','title', 'similarity']].values])
    display(HTML(h))
#example
recommendation('bd667dbd5200c9f07fb07cef29435f7ca7c2639b', k=5)
task = ["Effectiveness of drugs being developed and tried to treat COVID-19 patients.",
"Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.",
      ' '.join(keywords)]

tasks={'What do we know about vaccines and therapeutics?': task}
def relevant_articles(tasks, k=3, lower=1950, upper=2020, only_covid19=False):
    relevant_articles_dict = {}
    tasks = [tasks] if type(tasks) is str else tasks 
    
    tasks_vectorized = vectorizer.transform(tasks)
    tasks_topic_dist = pd.DataFrame(lda.transform(tasks_vectorized))

    for index, bullet in enumerate(tasks):
        print(bullet)
        recommended = get_k_nearest_docs(tasks_topic_dist.iloc[index], k, lower, upper, only_covid19)
        recommended = df.iloc[recommended]
        relevant_articles_dict.update({bullet: recommended})
    
    return relevant_articles_dict
relevant_dict = relevant_articles(task, 5, only_covid19=False)
def add_summary(simmilar_dict):

    model = Summarizer()
    for (key, df) in tqdm(simmilar_dict.items()):
        df['summary'] = df['body_text'].progress_apply(lambda x: model(x, min_length=10, max_length = 200))
        simmilar_dict.update({key: df})
    return simmilar_dict
def display_results(relevant_dict):
    for key, df in relevant_dict.items():
        print(key)
        h = '<br/>'.join(['<a href="' + l + '" target="_blank">'+ n + '</a>' + '<p>' + s + '</p>' for l, n, s in df[['url','title', 'abstract']].values])
        display(HTML(h))
display_results(relevant_dict)
# loading if needed
fl = open("../../data/processed/df_filtered_threshold-04_keywords-all.pkl", "rb")
df_keywords_th_04 = pickle.load(fl)
df_keywords_th_04.shape
from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords
def all_summary(df):
    all_txt = ' '.join(df["body_text"])
    summary = summarize(all_txt, word_count = 1000)
    return summary
all_summary(df_keywords_th_04)