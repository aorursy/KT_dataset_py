!pip install pandarallel

!pip install langdetect

!pip install glove_python

!pip install pyjarowinkler
# Basics

import gc

import re

import os

import json

import heapq

import pickle

import string

import random

import logging

import datetime

from tqdm import tqdm

from pathlib import Path

from copy import deepcopy

from unicodedata import normalize



# Multiprocessing, compute acceleration

import numba

from numba import prange

from pandarallel import pandarallel

from multiprocessing import Pool, cpu_count



# NLP

import nltk

import spacy

from langdetect import detect

from glove import Corpus, Glove

from gensim.utils import deaccent

# nltk.download('stopwords')

from nltk.corpus import stopwords

from gensim.models import CoherenceModel, TfidfModel

from gensim.models import LdaModel, LdaMulticore

from gensim.models.phrases import Phrases, Phraser

from gensim.corpora import Dictionary

from pyjarowinkler import distance



# Graphs

import networkx as nx



# Data Science

import umap

import numpy as np

import pandas as pd

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity



# Visuals

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook

import pyLDAvis

import pyLDAvis.gensim

from wordcloud import WordCloud, STOPWORDS



%matplotlib inline

output_notebook()
# Set figure size for matplotlib

plt.rcParams["figure.figsize"] = (10, 10)

pd.options.display.max_colwidth=160
# Install language models

!python -m spacy download en

!python -m spacy download fr

!python -m spacy download es

!python -m spacy download it

!python -m spacy download de

!python -m spacy download pt
class DataHandler(object):

    """

    Class that is used to load all the data from the CORD-19 dataset

    """

    def __init__(self, datapath):

        self.datapath = datapath



    def read_metadata(self, metadata_fileneme):

        """

        Read metadata csv

        """

        return pd.read_csv(self.datapath + metadata_fileneme)



    def read_papers(self, filelist):

        """

        Read json files from list of files

        """

        list_json_data = []



        for filename in filelist:

            file = json.load(open(filename, "rb"))

            list_json_data.append(file)



        return list_json_data



    def read_files_paths(self, folderpath, fileext=".json"):

        """

        Get paths to all the files inside a folder recursively. Set an extensions if needed

        """

        fls = [

            os.path.join(root, fn)

            for root, dirs, files in os.walk(Path(folderpath))

            for fn in files

            if Path(fn).suffix == fileext

        ]



        return fls



    def parse_authors(self, authors):

        """

        Parse authors field

        """

        authors_ls = []

        names_ls = []

        affiliations_ls = []

        emails_ls = []



        for author in authors:



            author_text = []



            # Parse name

            middle_name = " ".join(author["middle"])

            full_name = (

                " ".join([author["first"], middle_name, author["last"]])

                if author["middle"]

                else " ".join([author["first"], author["last"]])

            )

            author_text.append(full_name)

            names_ls.append(full_name)



            # Parse affiliation

            affiliation = author["affiliation"]

            if affiliation:

                affiliation_text = []

                laboratory = affiliation["laboratory"]

                institution = affiliation["institution"]

                location = affiliation["location"]

                if laboratory:

                    affiliation_text.append(laboratory)

                if institution:

                    affiliation_text.append(institution)

                if location:

                    affiliation_text.append(" ".join(list(location.values())))



                affiliation_text = ", ".join(affiliation_text)

                author_text.append(f"({affiliation_text})")

                affiliations_ls.append(affiliation_text)



            # Parse email

            email = author["email"]

            if email:

                author_text.append(f"[{email}]")

                emails_ls.append(email)



            # Concat info

            author_text = ", ".join(author_text)

            authors_ls.append(author_text)



        return {

            "authors_full": "; ".join(authors_ls),

            "names": "; ".join(names_ls),

            "emails": "; ".join(emails_ls),

            "affiliations": "; ".join(affiliations_ls),

        }



    def parse_abstract_body(self, body_text, raw_mode=False):

        """

        Parse abstract and body fields

        """

        entries_ls = []



        for entry in body_text:

            # Get section and its text

            text = entry["text"]

            section = entry["section"]



            if raw_mode != True:

                entries_ls.append(section)

                entries_ls.append(text)

            else:

                entries_ls.append({"text": text, "section": section})



        if raw_mode != True:

            return "\n\n".join(entries_ls)

        else:

            return entries_ls



    def parse_bib(self, bibs):

        """

        Parse bibliography field

        """

        if type(bibs) == dict:

            bibs = list(bibs.values())

        bibs = deepcopy(bibs)



        bibs_clean = []

        for bib in bibs:

            title = bib["title"]

            authors = bib["authors"]

            year = bib["year"]

            venue = bib["venue"]

            bibs_clean.append(f"{title} [{year}, {venue}]")



        return "; ".join(bibs_clean)



    def build_df(self, subfolder_papers, no_full_text=False):

        """

        Build a final Dataframe

        """

        list_json = self.read_files_paths(self.datapath + subfolder_papers)

        list_json_data = self.read_papers(list_json)



        raw_data = {

            "paper_id": [],

            "title": [],

            "authors": [],

            "authors_names": [],

            "authors_affiliations": [],

            "authors_emails": [],

            "abstract": [],

            "text": [],

            "bibliography": [],

            "raw_bibliography": [],

        }

        for file in tqdm(list_json_data):

            

            raw_data["paper_id"].append(file["paper_id"])

            raw_data["title"].append(file["metadata"]["title"])



            authors = self.parse_authors(file["metadata"]["authors"])

            raw_data["authors"].append(authors["authors_full"])

            raw_data["authors_names"].append(authors["names"])

            raw_data["authors_affiliations"].append(authors["affiliations"])

            raw_data["authors_emails"].append(authors["emails"])

            

            if "abstract" in file:

                raw_data["abstract"].append(self.parse_abstract_body(file["abstract"]))

            else:

                raw_data["abstract"].append("")

            if no_full_text == True:

                raw_data["text"].append("")

            else:

                raw_data["text"].append(self.parse_abstract_body(file["body_text"]))

            raw_data["bibliography"].append(self.parse_bib(file["bib_entries"]))

            raw_data["raw_bibliography"].append(file["bib_entries"])



        df_data = pd.DataFrame(raw_data)



        return df_data
data_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_file_path = "metadata.csv"
datahandler = DataHandler(data_path)
df_meta = datahandler.read_metadata(metadata_file_path)
df_biorxiv = datahandler.build_df('biorxiv_medrxiv')

df_biorxiv['subset'] = "biorxiv"
# df_comm = datahandler.build_df('comm_use_subset')

# df_comm['subset'] = "comm"
df_noncomm = datahandler.build_df('noncomm_use_subset')

df_noncomm['subset'] = "noncomm"
# df_custom = datahandler.build_df('custom_license')

# df_custom['subset'] = "custom_license"
columns_to_keep = [

    "paper_id",

#     "source_x",

#     "journal",

#     "doi",

#     "pmcid",

#     "pubmed_id",

    "title_x",

    "authors_x",

#     "authors_names",

#     "authors_affiliations",

#     "authors_emails",

    "abstract_x",

    "publish_time",

    "text",

    "bibliography",

    "raw_bibliography",

#     "license",

#     "Microsoft Academic Paper ID",

#     "WHO #Covidence",

    "subset"

]
# Declare the datasets you want to merge

datasets_to_read = [

    df_biorxiv,

    df_noncomm

]
# Concat dfs generated from jsons, drop duplicates according to paper_id and abstract

df_all = pd.concat(datasets_to_read)

df_all = df_all.drop_duplicates(subset=['paper_id', 'abstract',]).reset_index(drop=True)
# Merge raw data with metadata

df_merged = pd.merge(df_meta, df_all, left_on='sha', right_on='paper_id', how='inner')[columns_to_keep]

# Rename columns

df_merged = df_merged.rename(columns={'source_x': 'source', 'title_x': 'title', 'authors_x': 'authors', 'abstract_x': 'abstract',})

# Replace empty string fields with nans

df_merged = df_merged.replace(r'^\s*$', np.nan, regex=True)
# Drop duplicates again and reset index

df_merged.drop_duplicates(subset='abstract', inplace=True)

df_merged.reset_index(drop=True, inplace=True)
# Create column with all text data

df_merged['title_abstract_text'] = df_merged['title'].astype(str) + "\n\n" + df_merged['abstract'].astype(str) + "\n\n" + df_merged['text'].astype(str)
# Check for nans

df_merged.loc[:, df_merged.isnull().any()].columns
df_merged['authors'].fillna("", inplace=True)

df_merged['abstract'].fillna("", inplace=True)
df_merged.shape
class Preprocessor(object):

    """

    Class that is used to clean raw str

    """

    def __init__(self, force_deaccent, force_ascii, min_token_len, stop_list_custom, disabled_components, default_language):

        

        self.force_deaccent = force_deaccent

        self.force_ascii = force_ascii

        self.default_language = default_language

        self.min_token_len = min_token_len

        self.nlp_list = {

            "en": spacy.load("en", disable=disabled_components),

            "fr": spacy.load("fr", disable=disabled_components),

            "es": spacy.load("es", disable=disabled_components),

            "it": spacy.load("it", disable=disabled_components),

            "de": spacy.load("de", disable=disabled_components),

            "pt": spacy.load("pt", disable=disabled_components),

            }

        self.stop_list_all = self.init_stopwords(stop_list_custom)



    def init_stopwords(self, stop_list_custom):

        """

        Get list of stopwords from nltk, spacy and used custom list 

        """

        # Get nltk stopwords

        stop_list_nltk = list(set(stopwords.words('english'))) \

        + list(set(stopwords.words('french'))) \

        + list(set(stopwords.words('spanish'))) \

        + list(set(stopwords.words('italian'))) \

        + list(set(stopwords.words('german'))) \

        + list(set(stopwords.words('portuguese')))

        # Get spacy stopwords

        stop_list_spacy = list(spacy.lang.en.stop_words.STOP_WORDS) \

        + list(spacy.lang.fr.stop_words.STOP_WORDS) \

        + list(spacy.lang.es.stop_words.STOP_WORDS) \

        + list(spacy.lang.it.stop_words.STOP_WORDS) \

        + list(spacy.lang.de.stop_words.STOP_WORDS) \

        + list(spacy.lang.pt.stop_words.STOP_WORDS) \

        

        return stop_list_nltk+stop_list_spacy+stop_list_custom

    

    def detect_language(self, text):

        """

        Detect language with langdetect

        """

        try:

            lang = detect(text)

        except:

            lang = "unknown"

        return lang 



    def preprocess(self, text):

        """

        Main function to preprocess the text

        """

        lang = self.detect_language(text)

            

        if lang in ["en", "fr", "es", "it", "de", "pt"]:

            nlp = self.nlp_list[lang]

        else:

            nlp = self.nlp_list[self.default_language]

            

        # Delete some punctuation before preprocessing BUT not all of it because some can be involved in n-grams (e.g. "-")

        text=re.sub(r'[!"#$%&\'()*+,./:;<=>?@\[\\\]^_`{|}~]',r' ',text) 

        

        # Apply spacy to the text

        doc = nlp(text)

        # Lemmatization, remotion of noise (stopwords, digit, puntuaction and singol characters)

        tokens = [

            token.lemma_ for token in doc if

            token.lemma_ != '-PRON-'

            and not token.is_punct

            and not token.is_digit

            and not token.like_num

            and not token.like_url

            and not token.like_email

            and len(token.lemma_) >= self.min_token_len and len(token.text) >= self.min_token_len

            and token.lemma_.lower() not in self.stop_list_all and token.text.lower() not in self.stop_list_all

        ]

        

        # Recreation of the text

        text = " ".join(tokens)



        # Remove accents, normalize to ascii

        if self.force_ascii:

            text = normalize('NFD', text).encode('ascii', 'ignore').decode('UTF-8')

        

        if self.force_deaccent:

            text = deaccent(text)

    

        # Remove double spaces

        text=re.sub(r'\s+',r' ',text)

        

        # Set as lowercase

        text = text.lower().strip()



        return text
preprocessor = Preprocessor(

    force_ascii=False,

    force_deaccent=True,

    min_token_len=1,

    stop_list_custom=[

        'positives', 'true', 'false', 'tp', 'fp' 'cc_nc', 'q_q', 'r', 'b', 'p', 'q', 'h', 'cc', 'doi', 'medrxiv', 'fig', 'org', 'tb'

    ],

    disabled_components=['parser', 'ner'],

    default_language='en'

)
pandarallel.initialize(nb_workers=cpu_count(), progress_bar=True)
df_merged["title_abstract_text_preprocessed"] = df_merged.parallel_apply(

        lambda x: preprocessor.preprocess(x["title_abstract_text"]), axis=1

    )
def find_num_topics(dictionary, corpus, docs, end, start=2, step=2):

    """

    Train multiple LDA models in an indicated range of number of topics  

    """

    coherence_values = []

    model_list = []

    for num_topics in tqdm(range(start, end, step)):

        model = LdaModel(

            corpus=corpus,

            num_topics=num_topics,

            id2word=id2word,

            update_every=1,

            eval_every = 100,

            random_state=100,

            chunksize=2000,

            passes=4,

            iterations=100,

            per_word_topics=True,

        )



        model_list.append(model)

        coherencemodel = CoherenceModel(

            model=model, texts=docs, dictionary=dictionary, coherence="c_v"

        )

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
def plot_coherence_score(num_topic, coherence_score):

    """

    Plot coherence scores for LDA model

    """

    p = figure(plot_width=400, plot_height=400)



    # Add both a line and circles on the same plot

    p.line(num_topic, coherence_score, line_width=2)

    p.circle(num_topic, coherence_score, fill_color="white", size=8)

    p.xaxis.axis_label = "Number Of Topics"

    p.yaxis.axis_label = "Coherence Score"



    show(p)
def visualize_topics(lda_model, corpus, id2word):

    """

    Generate a visual dashboard of LDA topics

    """

    pyLDAvis.enable_notebook()

    

    return pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
def print_topics(lda_model):

    """

    Print topics made of keywords found in LDA model

    """

    columns_name = []

    pd_dict = {}



    num_topic = lda_model.num_topics

    for i in range(num_topic):

        columns_name.append(((f"Topic_{i+1}", "Word")))

        words, weight = zip(*lda_model.show_topic(i))

        pd_dict[f"topic{i+1}"] = list(words)



    df = pd.DataFrame(pd_dict)



    return df
# Split to tokens

docs_tokens = [doc.split() for doc in list(df_merged.title_abstract_text_preprocessed)]
# Infer bigram model

bigram_mod = Phraser(Phrases(docs_tokens, min_count=10, threshold=10))
# Save model if needed

# bigram_mod.save('./data/bigram_mod')
# Get bigrams

docs_bigram_tokens = [bigram_mod[doc] for doc in docs_tokens]
# Create Dictionary

id2word = Dictionary(docs_bigram_tokens)

id2word.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
# Save model if needed

# id2word.save('./data/id2word')
# Term Document Frequency

bow_corpus = [id2word.doc2bow(doc) for doc in docs_bigram_tokens]
# TF-IDF Document Frequency

tfidf = TfidfModel(bow_corpus)

tfidf_corpus = tfidf[bow_corpus]
start_topic, end_topic, step = 10, 18, 2



# Train LDA models

model_list, coherence_values = find_num_topics(

    dictionary=id2word,

    corpus=bow_corpus,

    docs=docs_bigram_tokens, 

    start=start_topic, 

    end=end_topic,

    step=step

    )
plot_coherence_score(range(start_topic, end_topic, step), coherence_values)
bow_lda_model = model_list[1]
# Compute Perplexity

print('Perplexity: ', bow_lda_model.log_perplexity(bow_corpus))
# visualize_topics(bow_lda_model, bow_corpus, id2word)
print_topics(bow_lda_model)
# Save model if needed

# bow_lda_model.save('bow_lda_model')
start_topic, end_topic, step = 10, 18, 2



tfidf_model_list, tfidf_coherence_values = find_num_topics(

    dictionary=id2word,

    corpus=tfidf_corpus,

    docs=docs_bigram_tokens,

    start=start_topic,

    end=end_topic,

    step=step

    )
plot_coherence_score(range(start_topic, end_topic, step), tfidf_coherence_values)
tfidf_lda_model = tfidf_model_list[0]
# Compute Perplexity

print('Perplexity: ', tfidf_lda_model.log_perplexity(tfidf_corpus))  
# visualize_topics(tfidf_lda_model, tfidf_corpus, id2word)
print_topics(tfidf_lda_model).style.background_gradient(cmap='viridis')
# Save model if needed

# tfidf_lda_model.save('tfidf_lda_model')
lda_model = bow_lda_model
def topic_all_documents(lda_model, corpus, texts):

    """

    Obtain topics for all the documents

    """

    # Init output

    documents_topic_df = pd.DataFrame()

    columns_name = ["Document", "Dominant Topic", "Topic Score", "Word List"]

    columns_name.extend([f"Topic_{i+1}" for i in range(lda_model.num_topics)])

    columns_name.append("Text")



    words_topic = []

    for i in range(lda_model.num_topics):

        x, _ = zip(*lda_model.show_topic(i))

        words_topic.append(list(x))



    for document_indx, topic_score in enumerate(lda_model.get_document_topics(corpus)):

        dominant_topic = sorted(topic_score, key=lambda x: x[1], reverse=True)[0]



        row_score = np.zeros(lda_model.num_topics)

        index, score = zip(*topic_score)

        row_score[list(index)] = score

        row_score = np.around(row_score, 4)



        documents_topic_df = documents_topic_df.append(

            pd.concat(

                [

                    pd.Series(

                        [

                            int(document_indx),

                            dominant_topic[0] + 1,

                            round(dominant_topic[1], 4),

                            words_topic[dominant_topic[0]],

                        ]

                    ),

                    pd.Series(row_score),

                ],

                ignore_index=True,

            ),

            ignore_index=True,

        )



    # Add original text to the end of the output

    contents = pd.Series(texts)

    documents_topic_df = pd.concat([documents_topic_df, contents], axis=1)



    documents_topic_df.columns = columns_name

    documents_topic_df["Dominant Topic"] = pd.to_numeric(

        documents_topic_df["Dominant Topic"]

    )

    documents_topic_df["Document"] = pd.to_numeric(documents_topic_df["Document"])



    return documents_topic_df
%%time

documents_topic_df = topic_all_documents(

    lda_model=lda_model, corpus=bow_corpus, texts=docs_bigram_tokens

)
documents_topic_df.head(5)
most_representative_df = pd.DataFrame()

domiant_topic_df = documents_topic_df.groupby("Dominant Topic")



for i, grp in domiant_topic_df:

    most_representative_df = pd.concat(

        [

            most_representative_df,

            grp.sort_values(["Topic Score"], ascending=False).head(1),

        ],

        axis=0,

    )



most_representative_df.reset_index(drop=True, inplace=True)



most_representative_df = most_representative_df.iloc[:, 0:4]



most_representative_df.columns = [

    "Document",

    "Topic_Num",

    "Best Topic Score",

    "Word List",

]
most_representative_df.style.background_gradient(cmap='viridis')
def topic_distribution(lda_model, documents_topic_df):

    """

    

    """

    topic_avg = documents_topic_df.groupby(['Dominant Topic'])["Topic Score"].mean()

    topic_count = documents_topic_df.groupby(['Dominant Topic'])['Document'].count()



    topic_df = pd.DataFrame()

    topic_df['Average'] = topic_avg

    topic_df['Count'] = topic_count

    topic_df = topic_df.reset_index()

    

    topic_df = topic_df.fillna(0)



    topic_df.plot.bar(x='Dominant Topic', y='Count', rot=0)



    return topic_df
topic_distibution_res = topic_distribution(lda_model, documents_topic_df)
topic_distibution_res.style.background_gradient(cmap='viridis')
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  



cloud = WordCloud(prefer_horizontal=1.0)



topics = lda_model.show_topics(formatted=False)



fig, axes = plt.subplots(4, 2, figsize=(10,10), sharex=True, sharey=True)



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
def plot_tsne(lda_model):    

    """

    Plot Documents Clusters based on topics scors

    """

    topic_score = documents_topic_df.iloc[:,4: 4 + lda_model.num_topics]

    

    topic_num = np.array(documents_topic_df.iloc[:,1]).astype(int)



    # tSNE Dimension Reduction

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

    tsne_lda = tsne_model.fit_transform(topic_score)



    # Plot the Topic Clusters using Bokeh

    output_notebook()

    n_topics = lda_model.num_topics

    mycolors = np.array([color for name, color in mcolors.CSS4_COLORS.items()])

    

    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 

                  plot_width=900, plot_height=700)

    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num + 5])

    show(plot)
plot_tsne(lda_model)
def prediction_unseen_doc(lda_model, doc, threshold=0.1):

    """

    Get the most representative topic of a new documment

    """

    doc_preprocessed = doc.split()

    doc_tokens = bigram_mod[doc_preprocessed]

    bow_tokens = id2word.doc2bow(doc_tokens)



    rows = []

    for i, score in sorted(

        lda_model.get_document_topics(bow_tokens), key=lambda x: x[1], reverse=True

    ):

        if score > threshold:

            words, _ = zip(*lda_model.show_topic(i))

            rows.append([f"Topic_{i+1}", score, "; ".join(words)])

            break



    return pd.DataFrame(rows, columns=["Topic", "Score", "Words"])
def document_same_topic(df_topic, documents_topic_df, df_merged):

    """

    Obtain documents that have the same topic as df_topic

    """



    for index, row in df_topic.iterrows():

        topic = int(row["Topic"].split("_")[-1])



        doc_same_topic = list(

            documents_topic_df[documents_topic_df["Dominant Topic"] == topic][

                "Document"

            ]

        )



        doc_detail = df_merged.loc[doc_same_topic]



    return doc_detail
# Define the queries

QUERY1 = '''

Data on potential risks factors

Smoking, pre-existing pulmonary disease

Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities

Neonates and pregnant women

Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.

'''

QUERY2 = '''

Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors

'''

QUERY3 = '''

Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups

'''

QUERY4 = '''

Susceptibility of populations

'''

QUERY5 = '''

Public health mitigation measures that could be effective for control

'''
# Preprocess the queries

query_1_preprocessed = preprocessor.preprocess(QUERY1)

query_2_preprocessed = preprocessor.preprocess(QUERY2)

query_3_preprocessed = preprocessor.preprocess(QUERY3)

query_4_preprocessed = preprocessor.preprocess(QUERY4)

query_5_preprocessed = preprocessor.preprocess(QUERY5)
unseen_doc_q1 = prediction_unseen_doc(lda_model=lda_model,doc=query_1_preprocessed)

unseen_doc_same_topic_q1 = document_same_topic(unseen_doc_q1, documents_topic_df, df_merged).head(10)
unseen_doc_q1.style.background_gradient(cmap='viridis')
unseen_doc_same_topic_q1.head()
unseen_doc_q2 = prediction_unseen_doc(lda_model=lda_model,doc=query_2_preprocessed)

unseen_doc_same_topic_q2 = document_same_topic(unseen_doc_q2, documents_topic_df, df_merged).head(10)
unseen_doc_q2.style.background_gradient(cmap='viridis')
unseen_doc_same_topic_q2.head()
unseen_doc_q3 = prediction_unseen_doc(lda_model=lda_model,doc=query_3_preprocessed)

unseen_doc_same_topic_q3 = document_same_topic(unseen_doc_q3, documents_topic_df, df_merged).head(10)
unseen_doc_q3.style.background_gradient(cmap='viridis')
unseen_doc_same_topic_q3.head()
unseen_doc_q4 = prediction_unseen_doc(lda_model=lda_model,doc=query_4_preprocessed)

unseen_doc_same_topic_q4 = document_same_topic(unseen_doc_q4, documents_topic_df, df_merged).head(10)
unseen_doc_q4.style.background_gradient(cmap='viridis')
unseen_doc_same_topic_q4.head()
unseen_doc_q5 = prediction_unseen_doc(lda_model=lda_model,doc=query_5_preprocessed)

unseen_doc_same_topic_q5 = document_same_topic(unseen_doc_q5, documents_topic_df, df_merged).head(10)
unseen_doc_q5.style.background_gradient(cmap='viridis')
unseen_doc_same_topic_q5.head()
GLOVE_TRAIN_MODE = False
# Create a Glove object which will use the corpus matrix created above lines to create embeddings

glove = Glove(no_components=300, learning_rate=0.05)

if GLOVE_TRAIN_MODE:

    # Creating a corpus object

    corpus = Corpus()



    # Fit the corpus with a list of tokens

    corpus.fit(docs_tokens, window=10)



    # Fit glove embeddings and add dict to it

    glove.fit(corpus.matrix, epochs=30, no_threads=30, verbose=True)

    glove.add_dictionary(corpus.dictionary)

    

    # Save obtained model

    glove.save('opencovid_glove_300d.model')
def most_similar(v, *ignore, n=1):

    """

    Get most similar words using words embeddings and Glove vector matrix 

    """

    similar = []

    for word, u in vectors.items():

        if word in ignore:

            continue

        similarity = u.dot(v)

        if len(similar) < n:

            heapq.heappush(similar, (similarity, word))

        else:

            heapq.heappushpop(similar, (similarity, word))



    return sorted(similar, reverse=True)
def plot_wordsl(words, lines=False):

    """

    Plot simple words relations with Glove embeddings

    """

    BW = "\x1b[1;30;45m"

    EEND = "\x1b[0m"



    wwl = []

    for ww in words:

        if ww in vectors:

            wwl.append(ww)

        else:

            print(

                BW,

                "*** WARNING ***** the word ",

                ww,

                "is not in the embedding vectors",

                EEND,

            )



    words = wwl

    pca = PCA(n_components=2)

    xys = pca.fit_transform([vectors[w] for w in words])



    if lines:

        for i in range(0, len(words), 2):

            plt.plot(xys[i : i + 2, 0], xys[i : i + 2, 1])

    else:

        plt.scatter(*xys.T)



    for word, xy in zip(words, xys):

        plt.annotate(word, xy, fontsize=15)



    return pca
# If load from saved copy

glove = Glove.load('/kaggle/input/assystem-opencovid19-helpers/opencovid_glove_300d.model')
# Check embeddings dimensions

len(glove.word_vectors[glove.dictionary['virus']])
# Check number of words contained in dict

len(glove.dictionary)
# Put in a dictionary to easily get embeddings

vectors = {word: glove.word_vectors[glove.dictionary[word]] for word in glove.dictionary.keys()}

len(vectors)
print(vectors['virus'])
# Test it 1!

xxll = most_similar(vectors['covid19'], n=10)

near_w = [x[1] for x in xxll]

near_w
# Test it 2!

xxll = most_similar(vectors['pneumonia'], n=10)

near_w = [x[1] for x in xxll]

near_w
plt.title('Words relations using Glove', fontsize=20)

plot_wordsl(['pneumonia',

 'severe',

 'respiratory',

 'bacterial',

 'patient',

 'acute',

 'child',

 'bronchiolitis',

 'interstitial',

 'cause'], lines=True)
@numba.jit(target="cpu", nopython=True, parallel=True, fastmath=True)

def fast_cosine(u, v):

    """

    Compute cosine distance between two matrices

    """

    m = u.shape[0]

    udotv = 0

    u_norm = 0

    v_norm = 0

    for i in range(m):

        if (np.isnan(u[i])) or (np.isnan(v[i])):

            continue



        udotv += u[i] * v[i]

        u_norm += u[i] * u[i]

        v_norm += v[i] * v[i]



    u_norm = np.sqrt(u_norm)

    v_norm = np.sqrt(v_norm)



    if (u_norm == 0) or (v_norm == 0):

        ratio = 0.0

    else:

        ratio = udotv / (u_norm * v_norm)

    return ratio
@numba.jit(target="cpu", nopython=True, parallel=True, fastmath=True)

def fast_get_keyWordsMatching_compute(Xtf1, Xtf2):

    """

    Compute multiplication element by element of vectors and return index of non negative value after multiplication

    """

    pointwiseMultiplication = np.multiply(Xtf1, Xtf2)

    index = pointwiseMultiplication.nonzero()[1]



    return pointwiseMultiplication, index
def fast_get_keyWordsMatching(Xtf1, Xtf2, tf):

    """

    Retrieve keywords responsible of the matching score

    """

    pointwiseMultiplication, index = fast_get_keyWordsMatching_compute(

        Xtf1.toarray(), Xtf2.toarray()

    )

    cross = []

    for i in index:

        # take the word from the dictionary that corrisponds to the index

        key = list(tf.vocabulary_.keys())[list(tf.vocabulary_.values()).index(i)]

        # take the value of the multiplication

        value = pointwiseMultiplication[0, i]

        cross.append((key, value))



    return sorted(cross, key=lambda x: -x[1])  # sort output
def get_matching(df_merged, xtfidf_papers, xtfidf_query):

    """

    Compute scores for documents

    """

    score_overall_papers = []

    keywords_papers = []

    for i in prange(df_merged.shape[0]):

        score_overall_papers.append(

            fast_cosine(xtfidf_papers[i].toarray()[0], xtfidf_query.toarray()[0])

        )

        keywords_papers.append(

            fast_get_keyWordsMatching(xtfidf_papers[i], xtfidf_query, tfidf)

        )



    return score_overall_papers, keywords_papers
# Query expansion 1 with Glove

query_1_expanded_l = []

for ww in query_1_preprocessed.split():

    if ww not in ['neonates']:

        xxll = most_similar(vectors[ww], n=3)

        near_w = [x[1] for x in xxll]

        query_1_expanded_l += near_w

query_1_expanded = " ".join(query_1_expanded_l)
query_1_preprocessed, query_1_expanded
# Query expansion 2 with Glove

query_2_expanded_l = []

for ww in query_2_preprocessed.split():

    xxll = most_similar(vectors[ww], n=3)

    near_w = [x[1] for x in xxll]

    query_2_expanded_l += near_w

query_2_expanded = " ".join(query_2_expanded_l)
query_2_preprocessed, query_2_expanded
# Query expansion 3 with Glove

query_3_expanded_l = []

for ww in query_3_preprocessed.split():

    xxll = most_similar(vectors[ww], n=3)

    near_w = [x[1] for x in xxll]

    query_3_expanded_l += near_w

query_3_expanded = " ".join(query_3_expanded_l)
query_3_preprocessed, query_3_expanded
# Query expansion 4 with Glove

query_4_expanded_l = []

for ww in query_4_preprocessed.split():

    xxll = most_similar(vectors[ww], n=3)

    near_w = [x[1] for x in xxll]

    query_4_expanded_l += near_w

query_4_expanded = " ".join(query_4_expanded_l)
query_4_preprocessed, query_4_expanded
# Query expansion 5 with Glove

query_5_expanded_l = []

for ww in query_5_preprocessed.split():

    xxll = most_similar(vectors[ww], n=3)

    near_w = [x[1] for x in xxll]

    query_5_expanded_l += near_w

query_5_expanded = " ".join(query_5_expanded_l)
query_5_preprocessed, query_5_expanded
# Initialize TfIdfVectorizer object

tfidf = TfidfVectorizer(

        ngram_range=(1, 1)

)
# Fit TfIdf model

tfidf.fit(df_merged["title_abstract_text_preprocessed"])
# Get sparse matrices for papers 

xtfidf_papers = tfidf.transform(df_merged["title_abstract_text_preprocessed"])
# Get sparse matrices for queries

xtfidf_query_1 = tfidf.transform([query_1_expanded])

xtfidf_query_2 = tfidf.transform([query_2_expanded])

xtfidf_query_3 = tfidf.transform([query_3_expanded])

xtfidf_query_4 = tfidf.transform([query_4_expanded])

xtfidf_query_5 = tfidf.transform([query_5_expanded])
# Get matching scores and keywords that matched for queries

scores_1, keywords_1 = get_matching(df_merged, xtfidf_papers, xtfidf_query_1)

scores_2, keywords_2 = get_matching(df_merged, xtfidf_papers, xtfidf_query_2)

scores_3, keywords_3 = get_matching(df_merged, xtfidf_papers, xtfidf_query_3)

scores_4, keywords_4 = get_matching(df_merged, xtfidf_papers, xtfidf_query_4)

scores_5, keywords_5 = get_matching(df_merged, xtfidf_papers, xtfidf_query_5)
df_merged_q1 = df_merged[['paper_id', 'title']].copy()

df_merged_q2 = df_merged[['paper_id', 'title']].copy()

df_merged_q3 = df_merged[['paper_id', 'title']].copy()

df_merged_q4 = df_merged[['paper_id', 'title']].copy()

df_merged_q5 = df_merged[['paper_id', 'title']].copy()
df_merged_q1['score'] = scores_1

df_merged_q2['score'] = scores_2

df_merged_q3['score'] = scores_3

df_merged_q4['score'] = scores_4

df_merged_q5['score'] = scores_5

df_merged_q1['keywords'] = [[x[0] for x in r][:2] for r in keywords_1]

df_merged_q2['keywords'] = [[x[0] for x in r][:2] for r in keywords_2]

df_merged_q3['keywords'] = [[x[0] for x in r][:2] for r in keywords_3]

df_merged_q4['keywords'] = [[x[0] for x in r][:2] for r in keywords_4]

df_merged_q5['keywords'] = [[x[0] for x in r][:2] for r in keywords_5]
df_merged_q1 = df_merged_q1.sort_values(by='score', ascending=False).reset_index(drop=True)

df_merged_q2 = df_merged_q2.sort_values(by='score', ascending=False).reset_index(drop=True)

df_merged_q3 = df_merged_q3.sort_values(by='score', ascending=False).reset_index(drop=True)

df_merged_q4 = df_merged_q4.sort_values(by='score', ascending=False).reset_index(drop=True)

df_merged_q5 = df_merged_q5.sort_values(by='score', ascending=False).reset_index(drop=True)
# Cut too long title for pretty printing

df_merged_q1['title'] = df_merged_q1['title'].apply(lambda x: " ".join(x.split()[:10])+"...")

df_merged_q2['title'] = df_merged_q2['title'].apply(lambda x: " ".join(x.split()[:10])+"...")

df_merged_q3['title'] = df_merged_q3['title'].apply(lambda x: " ".join(x.split()[:10])+"...")

df_merged_q4['title'] = df_merged_q4['title'].apply(lambda x: " ".join(x.split()[:10])+"...")

df_merged_q5['title'] = df_merged_q5['title'].apply(lambda x: " ".join(x.split()[:10])+"...")
df_merged_q1.head(10).style.background_gradient(cmap='viridis')
df_merged_q2.head(10).style.background_gradient(cmap='viridis')
df_merged_q3.head(10).style.background_gradient(cmap='viridis')
df_merged_q4.head(10).style.background_gradient(cmap='viridis')
df_merged_q5.head(10).style.background_gradient(cmap='viridis')
QUERY_NEW = "Air pollution, arthritis"

query_new_preprocessed = preprocessor.preprocess(QUERY_NEW)
# Query expansion 4 with Glove

query_new_expanded_l = []

for ww in query_new_preprocessed.split():

    xxll = most_similar(vectors[ww], n=3)

    near_w = [x[1] for x in xxll]

    query_new_expanded_l += near_w

query_new_expanded = " ".join(query_new_expanded_l)
query_new_preprocessed, query_new_expanded
xtfidf_query_new = tfidf.transform([query_new_expanded])
scores_new, keywords_new = get_matching(df_merged, xtfidf_papers, xtfidf_query_new)
df_merged_qnew = df_merged[['paper_id', 'title']].copy()

df_merged_qnew['score'] = scores_new

df_merged_qnew['keywords'] = [[x[0] for x in r][:2] for r in keywords_new]
df_merged_qnew = df_merged_qnew.sort_values(by='score', ascending=False).reset_index(drop=True)

df_merged_qnew['title'] = df_merged_qnew['title'].apply(lambda x: " ".join(x.split()[:10])+"...")
df_merged_qnew.head(10).style.background_gradient(cmap='viridis')
def check_cite(df, dict_merged_title_bib, citations):

    """

    Generate graph nodes and edges

    """

    cite_frame = set()

    for cite in df['bibliography']:

        for dict_paper in dict_merged_title_bib:

            processed_title = dict_paper['title']

            if cite and processed_title:

                try:

                    if distance.get_jaro_distance(processed_title, cite, winkler=True, scaling=0.1) > 0.88:

                        citations.add((df['paper_id'],dict_paper['paper_id']))

                        cite_frame.add((df['paper_id'],dict_paper['paper_id']))

                except:

                    break

    return cite_frame
def clean_for_graphs(text):

    """

    Light prepossessing steps that are used to preprocess text for graph generation

    """

    # convert string to upper case

    text = text.lower()

    # prepare regex for char filtering

    re_print = re.compile("[^%s]" % re.escape(string.printable))

    # prepare translation table for removing punctuation

    table = str.maketrans("", "", string.punctuation)

    # normalize unicode characters

    text = normalize("NFD", text).encode("ascii", "ignore")

    text = text.decode("UTF-8")

    # tokenize on white space

    words = text.split()

    # remove punctuation from each token

    words = [w.translate(table) for w in words]

    # remove non-printable chars form each token

    words = [re_print.sub("", w) for w in words]

    new_text = " ".join(words)



    return new_text
GRAPH_TRAIN_MODE = False
df_merged_title_bib =  df_merged[['paper_id','title','bibliography']].copy()

if GRAPH_TRAIN_MODE:



    pandarallel.initialize(nb_workers=cpu_count()-1, progress_bar=True)

    

    df_merged_title_bib['bibliography'] = df_merged_title_bib['bibliography'].apply(lambda x: x.split(';'))

    

    dict_merged_title_bib = df_merged_title_bib.to_dict('records')

    df_merged_title_bib['title'] = df_merged_title_bib['title'].apply(clean_for_graphs)

    df_merged_title_bib['bibliography'] = df_merged_title_bib['bibliography'].apply(lambda x: [clean_for_graphs(cite) for cite in x])

    citations = set()

    df_merged_title_bib['cite'] = df_merged_title_bib.parallel_apply(check_cite, args=(dict_merged_title_bib, citations), axis=1)
# Load from Assystem-OpenCovid-Helpers Dataset if needed

df_merged_title_bib = pd.read_pickle('/kaggle/input/assystem-opencovid19-helpers/opencovid_df_bib_graph_1000.pkl')
# Get paper_id --> title matrix for prettyprinting

paper_id_to_title = df_merged_title_bib[['paper_id', 'title']].groupby('paper_id')['title'].apply(list).to_dict()

paper_id_to_title = {k:v[0] for k, v in paper_id_to_title.items()}
# Drop empty links

citations = list(set([list(i)[0] for i in list(df_merged_title_bib['cite']) if len(i)>0]))
# Create a Directed graph and add edges to it

DG=nx.DiGraph()

DG.add_edges_from(citations, color='red')
# Generate positions and insert to nodes attrs

pos = nx.random_layout(DG) 

for n, p in pos.items():

    DG.nodes[n]['pos'] = p.tolist()
edge_x = []

edge_y = []

for edge in DG.edges():

    x0, y0 = DG.nodes[edge[0]]['pos']

    x1, y1 = DG.nodes[edge[1]]['pos']

    edge_x.append(x0)

    edge_x.append(x1)

    edge_x.append(None)

    edge_y.append(y0)

    edge_y.append(y1)

    edge_y.append(None)
edge_trace = go.Scatter(

    x=edge_x, y=edge_y,

    line=dict(width=0.5, color='#888'),

    hoverinfo='none',

    mode='lines')



node_x = []

node_y = []

for node in DG.nodes():

    x, y = DG.nodes[node]['pos']

    node_x.append(x)

    node_y.append(y)



node_trace = go.Scatter(

    x=node_x, y=node_y,

    mode='markers',

    hoverinfo='text',

    marker=dict(

        showscale=True,

        colorscale='Viridis',

        reversescale=True,

        color=[],

        size=13,

        colorbar=dict(

            thickness=15,

            title='Node Connections',

            xanchor='left',

            titleside='right'

        ),

        line_width=2))
node_adjacencies = []

node_text = []

for node, adjacencies in dict(DG.degree()).items():

    node_adjacencies.append(adjacencies)

    node_text.append(f"Paper ID: {node}<br>Title: {paper_id_to_title[node]}<br>Nb of connections: {str(adjacencies)}")



node_trace.marker.color = node_adjacencies

node_trace.text = node_text
fig = go.Figure(data=[edge_trace, node_trace],

                layout=go.Layout(

                title='Network graph of a subset of COVID19 documents dataset (1000 samples)',

                titlefont_size=10,

                showlegend=False,

                hovermode='closest',

                margin=dict(b=20,l=5,r=5,t=40),

                annotations=[ dict(

                    text="COVID-19 Open Research Challenge",

                    showarrow=False,

                    xref="paper", yref="paper",

                    x=0.005, y=-0.002)],

                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),

                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

                )

fig.show()
dict_degree = dict(DG.degree)



fig = plt.gcf()

# fig.set_size_inches(20, 20)

pos = nx.layout.spring_layout(DG)

node_sizes = [v * 40 for v in dict_degree.values()]

M = DG.number_of_edges()

nodes = nx.draw_networkx_nodes(DG, pos, node_size=node_sizes, node_color='red')

edges = nx.draw_networkx_edges(DG, pos, node_size=node_sizes, arrowstyle='->',

                               edge_cmap=plt.cm.Blues)



plt.show()

# fig.savefig('citations_graph.png', dpi=100)
dict(list(dict(DG.in_degree).items())[:5])
sorted(DG.in_degree, key=lambda x: x[1], reverse=True)[:5]
dict(list(dict(DG.out_degree).items())[:5])
sorted(DG.out_degree, key=lambda x: x[1], reverse=True)[:5]
fig = plt.gcf()

# fig.set_size_inches(24, 18)

pos = nx.layout.spring_layout(DG)



dict_in_degree = dict(DG.in_degree)

dict_out_degree = dict(DG.out_degree)

node_sizes_in = [v * 40 for v in dict_in_degree.values()]

node_sizes_out = [v * 40 for v in dict_out_degree.values()]

M = DG.number_of_edges()

nodes = nx.draw_networkx_nodes(DG, pos, node_size=node_sizes_in, node_color='red')



nodes = nx.draw_networkx_nodes(DG, pos, node_size=node_sizes_out, node_color='blue')

edges = nx.draw_networkx_edges(DG, pos, node_size=node_sizes, arrowstyle='->',

                               edge_cmap=plt.cm.Blues)



plt.show()

# fig.savefig('citations_graph_in_out_degree.png', dpi=100)
fig = plt.gcf()

# fig.set_size_inches(24, 18)

pos = nx.layout.spring_layout(DG)



dict_in_degree = dict(DG.in_degree)

dict_out_degree = dict(DG.out_degree)

node_sizes_in = [v * 40 for v in dict_in_degree.values()]

node_sizes_out = [v * 40 for v in dict_out_degree.values()]

M = DG.number_of_edges()



Gcc = sorted(nx.weakly_connected_components(DG), key=len, reverse=True)

G_connected = DG.subgraph(Gcc[0])



node_connected = G_connected.nodes

edges_connected = G_connected.edges



node_unconnected =  DG.nodes - G_connected.nodes

edges_unconnected = DG.edges - G_connected.edges

G_unconnected=nx.DiGraph()

G_unconnected.add_edges_from(edges_unconnected, color='red')





dict_connected = dict(G_connected.degree)

dict_unconnected = dict(G_unconnected.degree)

node_sizes_connected = [v * 40 for v in dict_connected.values()]

node_sizes_unconnected = [v * 40 for v in dict_unconnected.values()]



nodes_connected_draw = nx.draw_networkx_nodes(G_connected, pos, node_color='red',node_size=node_sizes_connected )

edges_connected_draw = nx.draw_networkx_edges(G_connected, pos, node_size=node_sizes_connected,

                       arrowstyle='->',

                        edge_color= 'red',

                        edge_cmap=plt.cm.Blues

                      )



nodes_unconnected_draw = nx.draw_networkx_nodes(G_unconnected, pos, node_color='blue',node_size=node_sizes_unconnected)

edges_unconnected_draw = nx.draw_networkx_edges(G_unconnected, pos, node_size=node_sizes_unconnected,

                       arrowstyle='->',

                       edge_color= 'blue',                        

                       edge_cmap=plt.cm.Blues

                      )



plt.show()

# fig.savefig('connected_components_graph.png', dpi=100)
list(nx.weakly_connected_components(DG))[:1]
sorted(list(nx.weakly_connected_components(DG)), key=len, reverse=True)[:1]
list(nx.strongly_connected_components(DG))[:5]
sorted(list(nx.strongly_connected_components(DG)), key=len, reverse=True)[:5]
def build_dict_node(df):

    """

    Build a dictionary mapping each paper id to an index in 

    [0..num_of_paper - 1]

    """

    dict_node = dict()

    for i in range(len(df['paper_id'])):

        dict_node[df['paper_id'].iloc[i]] = i

    return dict_node
sub_graph = ['09b322d7bbb2bec7d12d4fb16d8c397118e33dad',

  '22c4892020511ff70f17fbaec468003b9287462e',

  '253af824466307bf9415fd7419f6029dc9180939',

  '8dbeac1f0bb7e99bc67744546f62dd4b199646cc',

  '93478a74599203ababc690436e54bb3ceb780061',

  'c715062c7f223c58fb278fb9470f084ec0b4f944']

df_merged_picked_6 = pd.read_pickle('/kaggle/input/assystem-opencovid19-helpers/df_merged_title_bib_picked_6.pkl')
# The edges

list(DG.edges(sub_graph))
nodes_sub_graph = list(sub_graph)
df_sub_graph =  df_merged_picked_6[df_merged_picked_6['paper_id'].isin(nodes_sub_graph)]
dict_sub_graph = build_dict_node(df_sub_graph)
dict_sub_graph
abstract = df_sub_graph['abstract']

abstract_list = abstract.to_list()
tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(abstract_list)
ddsim_matrix = cosine_similarity(tfidf_matrix[:], tfidf_matrix)
list_edges_sub_graph = list(DG.edges(list(sub_graph)))
def weight_edges(edges,matrix_tfidf,dict_node):

    """

    Generate weight edges

    """

    weight_edges = []

    for edge in edges:

        list_edge = list(edge)

        edge_start, edge_end = list_edge[:]

        weight_edge = matrix_tfidf[dict_node[edge_start]][dict_node[edge_end]]

        list_edge.append(weight_edge)

        weight_edges.append(tuple(list_edge))

    return weight_edges
list_edges_weighted_sub_graph = weight_edges(list_edges_sub_graph,ddsim_matrix,dict_sub_graph)
list_edges_weighted_sub_graph
directed_sub_graph = nx.DiGraph()
directed_sub_graph.add_weighted_edges_from(list_edges_weighted_sub_graph)
fig = plt.gcf()

fig.set_size_inches(15, 15)

pos = nx.spring_layout(directed_sub_graph)  # positions for all nodes

labels = nx.get_edge_attributes(directed_sub_graph,'weight')

dict_degree_sub_graph = dict(directed_sub_graph.degree)

node_sizes_sub_graph = [v * 80 for v in dict_degree_sub_graph.values()]

nx.draw_networkx_nodes(directed_sub_graph, pos

                       ,label=sub_graph, 

                       node_size=node_sizes_sub_graph, 

                       node_color="red")

nx.draw_networkx_edges(directed_sub_graph, pos, 

                       edgelist=list_edges_weighted_sub_graph,

                       width=1)



nx.draw_networkx_edge_labels(directed_sub_graph, pos, edge_labels=labels)





nx.draw_networkx_labels(directed_sub_graph, pos, edfont_size=8, font_family='sans-serif')

plt.axis('off')

plt.show()

# nx.draw_networkx_edge_labels(test_graph,pos=pos)
scores = np.random.random((1,6))[0].tolist()



df_sub_graph_id = df_sub_graph[['paper_id']]

df_sub_graph_id['score'] = scores 
df_sub_graph_id
nx.in_degree_centrality(directed_sub_graph)
degree_centrality = [ v for v in nx.in_degree_centrality(directed_sub_graph).values() ]
df_sub_graph_id['degree_centrality'] = degree_centrality
df_sub_graph_id.style.background_gradient(cmap='viridis')
df_sub_graph_id.set_index("paper_id", drop=True, inplace=True)
dict_score_sub_graph = df_sub_graph_id.to_dict(orient="index")
for key in dict_score_sub_graph.keys():

    dict_score_sub_graph[key]['neighbor'] = {}

    dict_score_sub_graph[key]['neighbor'] = dict(directed_sub_graph[key])
def dict_new_score_sub_graph(dict_score_sub_graph):

    for paper in dict_score_sub_graph:

        new_score = dict_score_sub_graph[paper]['score'] * dict_score_sub_graph[paper]['degree_centrality']

        for neighbor in dict_score_sub_graph[paper]['neighbor']:

            new_score = new_score + dict_score_sub_graph[paper]['neighbor'][neighbor]['weight'] * dict_score_sub_graph[neighbor]['score'] * dict_score_sub_graph[neighbor]['degree_centrality'] 

        dict_score_sub_graph[paper]['new_score'] = new_score

        del dict_score_sub_graph[paper]['neighbor']

    return dict_score_sub_graph
dict_new_score = dict_new_score_sub_graph(dict_score_sub_graph)
df_sub_graph_new_score = pd.DataFrame.from_dict(dict_new_score, orient='index')
df_sub_graph_new_score.style.background_gradient(cmap='viridis')