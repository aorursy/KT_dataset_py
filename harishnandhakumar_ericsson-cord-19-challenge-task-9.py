# Packages used within the scope of this analysis

!pip install rank_bm25

!pip install pyLDAvis

!pip install nltk 

!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz

!pip install spacy



!pip install scispacy

!pip install wordcloud

!pip install kneed

!pip install torch

!pip install sentence_transformers

!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

!pip install scattertext

!pip install gensim

!pip install yattag

!pip install bokeh

!pip install interact
# Modules that need to be imported for this analysis



import nltk

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download("punkt")

import pandas as pd

from pandas import ExcelWriter

from pandas import ExcelFile

import string

import re

import numpy as np



# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models.coherencemodel import CoherenceModel

from gensim.models.ldamodel import LdaModel

from gensim.corpora.dictionary import Dictionary

#Create Biagram & Trigram Models 

from gensim.models import Phrases



# spacy for lemmatization

import spacy



#sklearn

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



#bm25 imports

from rank_bm25 import BM25Okapi



#nltk

import nltk

from nltk.corpus import stopwords

from textblob import Word

from nltk.tokenize import RegexpTokenizer

from nltk.tokenize import word_tokenize, sent_tokenize , RegexpTokenizer

from nltk.stem import WordNetLemmatizer, PorterStemmer

from nltk.corpus import wordnet 

from wordcloud import WordCloud, STOPWORDS



from difflib import SequenceMatcher , get_close_matches, Differ

from sentence_transformers import SentenceTransformer

import scipy



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

%matplotlib inline



from kneed import KneeLocator

import matplotlib.colors as mcolors

import seaborn as sns



from pandas import Panel

from tqdm import tqdm_notebook as tqdm



# Enable logging for gensim - optional

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)



from pprint import pprint



import en_core_sci_md

import scattertext as st

import en_core_web_sm

from IPython.display import IFrame

# from kaggle.api.kaggle_api_extended import KaggleApi

import glob

import json



from IPython.display import HTML

from yattag import Doc, indent



from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure

from bokeh.layouts import column

from bokeh.models import RadioButtonGroup

from bokeh.models import TextInput

from bokeh.layouts import gridplot

from bokeh.models import Div

from bokeh.models import Paragraph

from bokeh.layouts import column, widgetbox

from gensim import corpora, models

from bokeh.io import output_notebook

from bokeh.plotting import figure, show

from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider

from bokeh.layouts import column

from bokeh.palettes import all_palettes

from ipywidgets import interact, interactive, fixed, interact_manual

import gc

import os

import pickle
# The files pulled from different directories 

# ## Metadatafile

meta_file = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

meta_df = pd.read_csv(meta_file)



# ## 4 json files

bio_path =  '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/'

comm_path = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/'

non_comm_path = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/'

custom_path = '/kaggle/input/CORD-19-research-challenge/custom_license/'
# Categories of journals

journals = {"BIORXIV_MEDRXIV": bio_path,

              "COMMON_USE_SUB" : comm_path,

              "NON_COMMON_USE_SUB" : non_comm_path,

              "CUSTOM_LICENSE" : custom_path}
# Function to parse each json file and merge abstract and body text into a new column 'full_text'



def parse_each_json_file(file_path,journal):

    inp = None

    with open(file_path) as f:

        inp = json.load(f)

    rec = {}

    rec['document_id'] = inp['paper_id'] or None

    rec['title'] = inp['metadata']['title'] or None

    if inp.get('abstract'):

        abstract = "\n ".join([inp['abstract'][_]['text'] for _ in range(len(inp['abstract']) - 1)])

        rec['abstract'] = abstract or None

    else:

        rec['abstract'] = None

    full_text = []

    for _ in range(len(inp['body_text'])):

        try:

            full_text.append(inp['body_text'][_]['text'])

        except:

            pass



    rec['full_text'] = "\n ".join(full_text) or None

    rec['source'] =  journal     or None    

    return rec
# Function to merge extracted data from json files into a pandas dataframe 



def parse_json_and_create_csv(journals):

    journal_dfs = []

    cnt = 0

    for journal, path in journals.items():

        print(journal,path)

        parsed_rcds = []  

        json_files = glob.glob('{}/**/*.json'.format(path), recursive=True)

        for file_name in json_files:

            cnt = cnt + 1

            #print('processing {} file {}'.format(cnt,file_name))

            rec = parse_each_json_file(file_name,journal)

            parsed_rcds.append(rec)

        print("Total Records in list = {}".format(len(parsed_rcds)))

        df = pd.DataFrame(parsed_rcds)

        journal_dfs.append(df)

        #print(journal_dfs)

    return pd.concat(journal_dfs)
# Create a csv file with extracted data from json files



all_df = parse_json_and_create_csv(journals=journals)



# Save the dataframe into a csv:

#all_df.to_csv("covid19_latest.csv",index=False)
# Drop Duplicates for the df from 4 json files based on document id and title keys and from metadata file

all_df = all_df.drop_duplicates(subset=['document_id', 'title'])



# Drop Duplicates for the meta data df based on sha key

meta_df = meta_df.drop_duplicates(subset=['sha'])
# Display dimensions of text and metadata dataframes

print('all_df:',all_df.shape)

print('meta_df:',meta_df.shape)
# Merging the useful columns from metadata file with the final_all_df file



covid_df = pd.merge(left=all_df, right=meta_df, how='left', left_on='document_id', right_on='sha')

covid_df['publish_time'] = covid_df['publish_time'].fillna('1900')

covid_df['publish_time'] = covid_df['publish_time'].str[:4]

covid_df['publish_time'] = covid_df['publish_time'].astype(int)

covid_df.fillna("",inplace=True)



### Saving this final merged dataframe into csv:covid10_final

#merged_df.to_csv("covid19_final.csv",index=False)
%%time

# Duplicate columns between the json files and meta data are dropped and column headers renamed



covid_df=covid_df.drop(columns=[ 'title_y', 'abstract_y','source_x','sha', 'Microsoft Academic Paper ID','license', 'WHO #Covidence', 'has_pdf_parse', 'has_pmc_xml_parse', 'full_text_file'])

covid_df=covid_df.rename(columns={"title_x":'title',"abstract_x":"abstract",'full_text':'body','document_id':'paper_id','source':'dataset'})



print('Dataset for before BM25 Scoring ',covid_df.shape)
print(covid_df.columns)

print('\n covid_df Shape:', covid_df.shape)



covid_df = covid_df[covid_df['body'].str.lower().str.contains('corona|sars|ncov|covid|ncovid|novel')]

print('\n covid_df after filtering Shape:', covid_df.shape)



covid_df= covid_df.drop_duplicates(subset=['title'])

print('\n covid_df after Title duplicate drop Shape:', covid_df.shape)



covid_df = covid_df.drop(['dataset', 'cord_uid', 'doi','pmcid', 'pubmed_id','journal'], axis = 1)

print('\n covid_df after columns drop Shape:', covid_df.columns)



covid_df = covid_df.reset_index(drop=True)
del [[meta_df,all_df]]

gc.collect()
# Pre-processing functions for cleaning text 



exclude_list = string.digits + string.punctuation

table = str.maketrans(exclude_list, len(exclude_list)*" ")

stop = stopwords.words('english')

english_stopwords = list(set(stop))

SEARCH_DISPLAY_COLUMNS = ['paper_id', 'title', 'body', 'publish_time', 'url', 'all_text']



nlp_x = en_core_web_sm.load()   



def clean_text(txt):    

    t = txt.replace("\\n",'')

    t = re.sub('\(|\)|:|,|;|\|’|”|“|\?|%|>|<', '', t )

    t = re.sub('/', ' ', t)

    t = t.replace('\n','')

    t = t.replace('  ','')

    t = t.replace("[",'')

    t = t.replace("]",'')

    t = ' '.join([word for word in t.split() if len(word)>1 ])

    t = sent_tokenize(t)

    return t



def preprocess_with_ngrams(docs):

    # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.

    bigram = Phrases(docs, min_count=5)

    trigram = Phrases(bigram[docs])



    for idx in range(len(docs)):

        for token in bigram[docs[idx]]:

            if '_' in token:

                # Token is a bigram, add to document.

                docs[idx].append(token)

        for token in trigram[docs[idx]]:

            if '_' in token:

                # Token is a trigram, add to document.

                docs[idx].append(token)

    return docs



class SearchResults:

    

    def __init__(self, 

                 data: pd.DataFrame,

                 columns = None):

        self.results = data

        if columns:

            self.results = self.results[columns]

            

    def __getitem__(self, item):

        return Paper(self.results.loc[item])

    

    def __len__(self):

        return len(self.results)

        

    def _repr_html_(self):

        return self.results._repr_html_()

    

    def getDf(self):        

        return self.results 

    

def strip_characters(text):

    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)

    t = re.sub('/', ' ', t)

    t = t.replace("'",'')

    return t



def clean(text):

    t = text.lower()

    t = strip_characters(t)

    t = str(t).translate(table)

    return t



def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words 

                     if len(word) > 1

                     and not word in english_stopwords

                     and not (word.isnumeric() and len(word) is not 4)

                     and (not word.isnumeric() or word.isalpha())] )

               )



def preprocess(text):

    t = clean(text)    

    tokens = tokenize(t)

    

    return tokens



# Functions defining the BM25 Algorithm



class WordTokenIndex:

    

    def __init__(self, 

                 corpus: pd.DataFrame, 

                 columns=SEARCH_DISPLAY_COLUMNS):

        self.corpus = corpus

        raw_search_str =self.corpus.title.fillna('') +' ' + self.corpus.body.fillna('')

        self.corpus['all_text'] = raw_search_str.apply(preprocess).to_frame()

        self.index = raw_search_str.apply(preprocess).to_frame()

        self.index.columns = ['terms']

        self.index.index = self.corpus.index

        self.columns = columns

       

    def search(self, search_string):

        search_terms = preprocess(search_string)

        result_index = self.index.terms.apply(lambda terms: any(i in terms for i in search_terms))

        results = self.corpus[result_index].copy().reset_index().rename(columns={'index':'paper'})

        return SearchResults(results, self.columns + ['paper'])

    

class RankBM25Index(WordTokenIndex):

    

    def __init__(self, corpus: pd.DataFrame, columns=SEARCH_DISPLAY_COLUMNS):

        super().__init__(corpus, columns)

        #self.bm25 = BM25Okapi(self.index.terms.tolist())

        self.bm25 = BM25Okapi(self.index.terms.tolist(),k1=3,b=0.001)

        

    def search(self, search_string, n=4):

        search_terms = preprocess(search_string)

        doc_scores = self.bm25.get_scores(search_terms)

        ind = np.argsort(doc_scores)[::-1][:n]

        results = self.corpus.iloc[ind][self.columns]

        results['BM25_Score'] = doc_scores[ind]

        results = results[results.BM25_Score > 0]

        return SearchResults(results.reset_index(), self.columns + ['BM25_Score'])

    

def show_task(taskTemp,taskId):

    #print(Task)

    keywords = taskTemp#tasks[tasks.Task == Task].Keywords.values[0]

    print(keywords)

    search_results = bm25_index.search(keywords, n=200)    

    return search_results
# Functions defining the BM25 Algorithm



class WordTokenIndex:

    

    def __init__(self, 

                 corpus: pd.DataFrame, 

                 columns=SEARCH_DISPLAY_COLUMNS):

        self.corpus = corpus

        raw_search_str =self.corpus.title.fillna('') +' ' + self.corpus.body.fillna('')

        self.corpus['all_text'] = raw_search_str.apply(preprocess).to_frame()

        self.index = raw_search_str.apply(preprocess).to_frame()

        self.index.columns = ['terms']

        self.index.index = self.corpus.index

        self.columns = columns

       

    def search(self, search_string):

        search_terms = preprocess(search_string)

        result_index = self.index.terms.apply(lambda terms: any(i in terms for i in search_terms))

        results = self.corpus[result_index].copy().reset_index().rename(columns={'index':'paper'})

        return SearchResults(results, self.columns + ['paper'])

    

class RankBM25Index(WordTokenIndex):

    

    def __init__(self, corpus: pd.DataFrame, columns=SEARCH_DISPLAY_COLUMNS):

        super().__init__(corpus, columns)

        #self.bm25 = BM25Okapi(self.index.terms.tolist())

        self.bm25 = BM25Okapi(self.index.terms.tolist(),k1=3,b=0.001)

        

    def search(self, search_string, n=4):

        search_terms = preprocess(search_string)

        doc_scores = self.bm25.get_scores(search_terms)

        ind = np.argsort(doc_scores)[::-1][:n]

        results = self.corpus.iloc[ind][self.columns]

        results['BM25_Score'] = doc_scores[ind]

        results = results[results.BM25_Score > 0]

        return SearchResults(results.reset_index(), self.columns + ['BM25_Score'])

    

def show_task(taskTemp,taskId):

    #print(Task)

    keywords = taskTemp#tasks[tasks.Task == Task].Keywords.values[0]

    print(keywords)

    search_results = bm25_index.search(keywords, n=200)    

    return search_results
%%time



rebuild_index = False



#BM25 algorithm getting trained on the text from the dataset



bm25index_file_create = '/kaggle/working/covid_task9_bm25index.pkl'

bm25index_file_load = '/kaggle/input/cord-19-bm25index/covid_task9_bm25index.pkl'

if rebuild_index:

        print("Running the BM25 index...")

        bm25_index = RankBM25Index(covid_df)

        print("Creating pickle file for the bm25 index...")

        with open(bm25index_file_create, 'wb') as file:

            pickle.dump(bm25_index, file)

        with open(bm25index_file_create, 'rb') as corpus_pt:

            bm25_index = pickle.load(corpus_pt)

        print("Completed load of the BM25 index from", bm25index_file_create, '...')

else:

    with open(bm25index_file_load, 'rb') as corpus_pt:

        bm25_index = pickle.load(corpus_pt)

    print("Completed load of the BM25 index from", bm25index_file_load, '...')





print("Shape of BM25: ", bm25_index.corpus.shape)
# LDA & Coherence functions



# Find the optimal topic model

def findOptimalTopicModel(start_num_topics, noOfMaxTopics, step_num_topics,model_list, coherence_values): 

    temp_coherenace = 0

    best_model_idx = 0

    idx = 0

    number_of_topics = 0

    x = range(start_num_topics, noOfMaxTopics, step_num_topics)

    for m, cv in zip(x, coherence_values):

        #print("Num Topics =", m, " has Coherence Value of", round(cv, 4))    

        if(temp_coherenace<cv):

            temp_coherenace = cv

            best_model_idx = idx

            number_of_topics = m

        idx += 1

 

    # Select the model and print the topics

    optimal_model = model_list[best_model_idx]

    model_topics = optimal_model.show_topics(num_topics=number_of_topics,formatted=False)

    return (optimal_model, model_topics, number_of_topics, temp_coherenace)



def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):

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

        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())        

    return model_list, coherence_values



def format_topics_sentences(df,ldamodel=None, corpus=None, texts=None):

    # Init output

    sent_topics_df = pd.DataFrame()

    

    # Get main topic in each documentp

    for i, row_list in enumerate(ldamodel[corpus]):

        row = row_list[0] if ldamodel.per_word_topics else row_list            

        

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic                

                wp = ldamodel.show_topic(topic_num)

                

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic_num', 'Topic_Perc_Contrib', 'Topic_Keywords']



    # Add original text to the end of the output

    #contents = pd.Series(texts)

    sent_topics_df = pd.concat([df,sent_topics_df], axis=1)

    return(sent_topics_df)
%%time

# BERT Sentence Transformer model based on bert-base-nli-mean-tokens



model = SentenceTransformer('bert-base-nli-mean-tokens')

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

top_N_sentence = 20

# BERT Functions for sentence embeddings



def sentenceEmbed(corpus_sentence, tcount):

    task = [task9_keywords[tcount]]

    corpus_sentence_embeddings = model.encode(corpus_sentence)

    query_embeddings = model.encode(task)



    for query, query_embedding in zip(task, query_embeddings):

        distances = scipy.spatial.distance.cdist([query_embedding], corpus_sentence_embeddings, "cosine")[0]



        results = zip(range(len(distances)), distances)

        results = sorted(results, key=lambda x: x[1])



        #print("\n\n======================\n\n")

        #print("Query:", task)

        #print("\nTop 5 most similar sentences in corpus:")

        all_sentence =''

        top_3_sentence=''

        score = 0

        top3 = 1

        #print(results[0:top_N_sentence])

        #print("\n\n======================\n\n")

        for idx, distance in results[0:top_N_sentence]:

            #print("\n\n",corpus_sentence[idx].strip(), "(Score: %.4f)" % (1-distance),"***END***")

            if(top3<4):                

                top_3_sentence += corpus_sentence[idx].strip().capitalize() +"--- \n\n"

                top3 += 1

            

            all_sentence += corpus_sentence[idx].strip()

            score += (1-distance)   

        score = score/top_N_sentence

        scoreStr =''

        if(score<=0.3):

            scoreStr = 'Low'

        elif(score>0.31 and score<0.7):

            scoreStr = 'Medium'

        else:

            scoreStr = 'High'

    return top_3_sentence,all_sentence,scoreStr;

# Function to determine Cut-off point for identifying optimum value



def kneeLocator(x,y):

    kn = KneeLocator(x, y,curve='concave',direction='decreasing',online=True)

    return y[kn.knee-1],kn;
# Plot of the curve that shows optimum cut-off value



def plotKnee(kn,x,y,minval,maxval,x_label,y_label,axes,idx):

    #plt.xlabel(x_label)

    #plt.ylabel(y_label)

    #plt.plot(x, y, 'bx-')

    #plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

    

    axes[idx].plot(x, y,'bx-')    

    axes[idx].vlines(kn.knee, minval,maxval, linestyles='dashed')

    axes[idx].set(xlabel =x_label, ylabel = y_label, title = 'Knee Scoring')



    #plt.show()

    
# Visualization Functions used to show Topic coherence and word cloud formed from dominant topics



def plotCoherence(start_num_topics, noOfMaxTopics, step_num_topics,coherence_values,axes,idx):

    x = range(start_num_topics, noOfMaxTopics, step_num_topics)

    axes[idx].plot(x, coherence_values)

    #plt.plot(x, coherence_values)

    #plt.xlabel("Num Of Topics")

    #plt.ylabel("Coherence score")

    #plt.legend(("coherence_values"), loc='best')

    axes[idx].set(xlabel ="Num Of Topics", ylabel = "Coherence score", title = 'Coherence Scoring')

    #plt.show()

    #return plt;



def plotWordCloud(number_of_topics,model_topics):

    if(number_of_topics>10):

        number_of_topics = 10

    N_rows = int(number_of_topics/2)

    N_cols = int(2)

    i = 0

    #Word cloud

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



    cloud = WordCloud(background_color='white',

                  width=2500,

                  height=1800,

                  max_words=10,

                  colormap='tab10',

                  color_func=lambda *args, **kwargs: cols[i],

                  prefer_horizontal=1.0)



    fig, axes = plt.subplots(N_rows,N_cols, figsize=(10,10), sharex=True, sharey=True)    

    for i, ax in enumerate(axes.flatten()):

        fig.add_subplot(ax)

        topic_words = dict(model_topics[i][1])    

        cloud.generate_from_frequencies(topic_words, max_font_size=300)

        plt.gca().imshow(cloud)

        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))

        plt.gca().axis('off')



    plt.subplots_adjust(wspace=0, hspace=0)

    plt.axis('off')

    plt.margins(x=0, y=0)

    plt.tight_layout()

    plt.show()

    return plt;
# Interactive visualizations including Scatterplot and TSNE 



def scatterPlot(df,fileName,topNStr):

    categoryName = ''

   

    if((df[df['SentenceScore'] == 'High']).shape[0] >0):

        categoryName = 'High'

    elif((df[df['SentenceScore'] == 'Medium']).shape[0] >0):

        categoryName = 'Medium'

    else:

        categoryName = 'Low'

    

    if(len(df.SentenceScore.unique()) >1):

        corpusOfNsentence = st.CorpusFromPandas(df,category_col="SentenceScore",text_col=topNStr,nlp=nlp_x).build()

        html = st.produce_scattertext_explorer(corpusOfNsentence,

                                        category=categoryName,

                                        category_name="Research Papers with "+categoryName+" Score",

                                        not_category_name='Others',

                                        width_in_pixels=1000,

                                        minimum_term_frequency=2,

                                        transform=st.Scalers.percentile)



        open(fileName, 'wb').write(html.encode('utf-8'))

        display(IFrame(src=fileName, width = 1800, height=700))

    else:

        print('No more than one category to produce scatter plot')



def tsneplot(df,topNStr):

    print(topNStr)



    tsne = TSNE(verbose=1, perplexity=5)

    np.random.seed(2017)

    texts = df['all_text'].values

    dictionary2 = corpora.Dictionary(texts)

    corpus2 = [dictionary2.doc2bow(text) for text in texts]



    ldamodel2 = models.ldamodel.LdaModel(corpus2, id2word=dictionary2, 

                                    num_topics=15, passes=20, minimum_probability=0)



    hm = np.array([[y for (x,y) in ldamodel2[corpus2[i]]] for i in range(len(corpus2))])

    tsne = TSNE(n_components=2)

    embedding = tsne.fit_transform(hm)

    embedding = pd.DataFrame(embedding, columns=['x','y'])

    embedding['hue'] = hm.argmax(axis=1)

    

    output_notebook()



    source = ColumnDataSource(

            data=dict(

            x = embedding.x,

            y = embedding.y,

            colors = [all_palettes['Category20'][15][i] for i in embedding.hue],

            title = df.title,

            year = df.publish_time,

            Top_Sentences = df[topNStr],

            SubTask = df.SubTask,

            url=df.url,

            alpha = [0.9] * embedding.shape[0],

            size = [14] * embedding.shape[0]

        )

    )

    hover_tsne = HoverTool(names=["final_results"], tooltips="""

        <div style="margin: 10">

            <div style="margin: 0 auto; width:300px;">

                <span style="font-size: 12px; font-weight: bold;">Title:</span>

                <span style="font-size: 12px">@title</span>

                <span style="font-size: 12px; font-weight: bold;">Year:</span>

                <span style="font-size: 12px">@year</span>

                <span style="font-size: 12px; font-weight: bold;">SubTask:</span>

                <span style="font-size: 12px">@SubTask</span>

                <span style="font-size: 12px; font-weight: bold;">URL:</span>

                <span style="font-size: 12px">@url</span>

            </div>

        </div>

        """)

    tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']

    plot_tsne = figure(plot_width=700, plot_height=700, tools=tools_tsne, title='Papers')

    plot_tsne.circle('x', 'y', size='size', fill_color='colors', 

                 alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="final_results")





    callback = CustomJS(args=dict(source=source), code=

    """

    var data = source.data;

    var f = cb_obj.value

    x = data['x']

    y = data['y']

    colors = data['colors']

    alpha = data['alpha']

    title = data['title']

    year = data['year']

    size = data['size']

    for (i = 0; i < x.length; i++) {

        if (year[i] <= f) {

            alpha[i] = 0.9

            size[i] = 7

        } else {

            alpha[i] = 0.05

            size[i] = 4

        }

    }

    source.change.emit();

    """)



    layout = column(plot_tsne)

    show(layout)
# HTML version of scatterplot



def generate_html_table(df):



    css_style = """table.paleBlueRows {

      font-family: "Trebuchet MS", Helvetica, sans-serif;

      border: 1px solid #FFFFFF;

      width: 100%;

      height: 150px;

      text-align: center;

      border-collapse: collapse;

    }

    table.paleBlueRows td, table.paleBlueRows th {

      text-align: center;

      border: 1px solid #FFFFFF;

      padding: 3px 2px;

      

    }

    table.paleBlueRows tbody td {

      text-align: center;

      font-size: 11px;

      

    }

    table.paleBlueRows tr:nth-child(even) {

      background: #D0E4F5;

    }

    table.paleBlueRows thead {

      background: #0B6FA4;

      border-bottom: 5px solid #FFFFFF;

    }

    table.paleBlueRows thead th {

      font-size: 17px;

      font-weight: bold;

      color: #FFFFFF;

      border-left: 2px solid #FFFFFF;

    }

    table.paleBlueRows thead th:first-child {

      border-left: none;

    }



    table.paleBlueRows tfoot {

      font-size: 14px;

      font-weight: bold;

      color: #333333;

      background: #D0E4F5;

      border-top: 3px solid #444444;

    }

    table.paleBlueRows tfoot td {

      font-size: 14px;

    }

    div.scrollable {width:100%; max-height:150px; overflow:auto; text-align: center;}

    """

    urlColIdx = df.columns.get_loc('url') 

    titleColIdx = df.columns.get_loc('title')

    pubColIdx = df.columns.get_loc('publish_time')

    

    doc, tag, text, line = Doc().ttl()



    with tag("head"):

        with tag("style"):

            text(css_style)





    with tag('table', klass='paleBlueRows'):

        with tag("tr"):

            for col in list(df.columns):

                if(col not in ('url')):

                    with tag("th"):

                         with tag("div", klass = "scrollable"):

                            text(col)

                        

        for idx, row in df.iterrows():

            with tag('tr'):

                for i in range(len(row)):

                    if(i==titleColIdx):                       

                        with tag('td'):

                            with tag("div", klass = "scrollable"):                            

                                if "http" in row[urlColIdx]:

                                    with tag("a", href = str(row[urlColIdx])):

                                        text(str(row[i]))

                                else:

                                    text(str(row[i]))

                    elif(i==pubColIdx):

                        with tag('td'):

                            with tag("div", klass = "scrollable"):                           

                                if(row[i]=="1900"):

                                    text("Not Available")

                                else:

                                    text(str(row[i]))

                    elif(i==urlColIdx):

                        None

                    else:

                        with tag('td'):

                            with tag("div", klass = "scrollable"):                            

                                text(str(row[i]))



    display(HTML(doc.getvalue()))
# This section is where we list text from the questions related to Task 9 of the COVID-19 challenge.

# Since Task 9 has multiple sub-questions, each is comma delimited so we can identify best responses for each individually.



task9_keywords = ["testing covid-19 sars-cov-2 coronavirus studies lab testing pandemic research data collection data standards nomenclature data gathering 2019-nCov SARS MERS",

"coronavirus insurance companies hospitals emergency room schools nursing homes workplaces covid-19 sars-cov-2 state officials local officials mitigation strategies telehealth colleges universities 2019-nCov MERS",

"coronavirus COVID19 COVID-19 SARS-CoV-2 at-risk  Understanding mitigating barriers information-sharing information sharing   information source insight discernment recognition shackles constraints hindrances impediments obstacles",

"coronavirus COVID19 COVID-19 SARS-CoV-2 recruit support coordinate local non-Federal expertise capacity relevant public health emergency response public private commercial non-profit academic",

"2020 2019 SARS-CoV-2 surveillance trace contact transmission public state interview evaluation monitor address interview symptoms",

"2020 2019 SARS-CoV-2 capacity interventions actionable prevent prepare funding investments financing public government future potential",

"aging population COVID19 Novel 2019 COVID-19 SARS-CoV-2 at-risk communication medical professionals critical workers relaying information social media",

"COVID19 Novel 2019 COVID-19 SARS-CoV-2 at-risk conveying information mitigation measures child care advice parents families children communications transparent protocol measures mitigation",

"risk disease population communications Coronavirus SARS-CoV-2 2019-nCov COVID-19 COVID19 COVID messaging notification contagion",

"misunderstanding containment mitigation misinterpretation regulation COVID-19 SARS-CoV-2 pathogens epidemiology coronavirus disease COVID19 2019",

"Action plan mitigate gaps problems inequity public health capability capacity funding citizens needs access surveillance treatment COVID19  2109 COVID-19 SARS-CoV-2",

"2020 2019 COVID-19 traditional approaches community-based interventions digital Inclusion develop local response ensuring communications marginalized disadvantaged populations research priorities",

"2020 2019 COVID-19 prison correctional federal state local inmate jail sheriff officer facility non-violent offenders guards deputy penal authorities locked incarcerated security custody",

"2020 2019 COVID-19 benefit patient client rejection insurance coverage care consultation eligibility plan risk factors policy therapy treatment payment in-network out-of-network deductible"]
task9 = ["1. Methods for coordinating data-gathering with standardized nomenclature.",

"2. Sharing response information among planners, providers, and others.",

"3. Understanding and mitigating barriers to information-sharing.",

"4. How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).",

"5. Integration of federal/state/local public health surveillance systems.",

"6. Value of investments in baseline public health response infrastructure preparedness",

"7. Modes of communicating with target high-risk populations (elderly, health care workers).",

"8. Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).",

"9. Communication that indicates potential risk of disease to all population groups.",

"10. Misunderstanding around containment and mitigation.",

"11. Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.",

"12. Measures to reach marginalized and disadvantaged populations. Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.",

"13. Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.",

"14. Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care"]
dict_task=dict(zip(task9,task9_keywords))
final_dict = {}

#header_dict = {'': '','All': 'All'}

header_dict = {'All': 'All'}

final_dict = dict(header_dict, **dict_task)
# Function that finds most relevent articles and sentences for a given input text. 



def bm25res(tcount,visualization):

    print('\033[1m' + '*********************************Start*****************************************')

    print('\033[1m' + 'Subtask: ' + task9[tcount] + '\n') 

    

    bm25_results = bm25_index.search(task9_keywords[tcount],n=covid_df.shape[0])

    bm25_df = bm25_results.getDf()

    

    bm25_score = bm25_df['BM25_Score'].sort_values(ascending=False).tolist()

    #print('Max BM25 Score = ',bm25_df['BM25_Score'].max())

    print('\033[1m' + 'BM25 Score Selection')

    bm25_x_idx = range(1, len(bm25_score)+1)    

    bm25_kn = KneeLocator(bm25_x_idx, bm25_score,curve='concave',direction='decreasing',online=True)

    #optimal_bm25_score ,kn_bm25= kneeLocator(bm25_x_idx,bm25_score)

    optimal_bm25_score = bm25_score[bm25_kn.knee-1]

    

    covid_df_bm25_filter =bm25_df[bm25_df['BM25_Score'] >=optimal_bm25_score]

    

    if (covid_df_bm25_filter.shape[0] <10):

        covid_df_bm25_filter= bm25_df.iloc[:25,:]



    print('\033[1m' + 'Number of Papers Selected after BM25 Scoring = ', covid_df_bm25_filter.shape[0])        

    docs = covid_df_bm25_filter.all_text

    dictionary = Dictionary(docs)

    if(len(docs)<10):

        dictionary.filter_extremes(no_below=1)

    else:

        dictionary.filter_extremes()

    #Create dictionary and corpus required for Topic Modeling

    corpus = [dictionary.doc2bow(doc) for doc in docs]

    noOfDocs = len(corpus)

    start_num_topics = 0

    step_num_topics = 2

    if(noOfDocs>=200):

        noOfMaxTopics = int(noOfDocs*0.1)

        if(noOfMaxTopics>100):

            noOfMaxTopics = 100

        start_num_topics = 5

        step_num_topics = 5

    elif(noOfDocs>=50 and noOfDocs<200):

        noOfMaxTopics = 20

        start_num_topics = 2

        step_num_topics = 2

    else:

        noOfMaxTopics = 10

        start_num_topics = 2

        step_num_topics = 2

    #print('Number of unique tokens: %d' % len(dictionary))

    #print('Number of documents: %d' % noOfDocs)

    #print('Number of max topics: %d' % noOfMaxTopics)

    print('\033[1m' + 'Finding Optimal of topics is in progress = ',noOfMaxTopics)

    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=start_num_topics, limit=noOfMaxTopics, step=step_num_topics)

    optimal_model, model_topics, number_of_topics, temp_coherenace= findOptimalTopicModel(start_num_topics, noOfMaxTopics, step_num_topics, model_list, coherence_values)

    print('\033[1m' + 'Optimal Number of Topics = ',number_of_topics)

    print('\033[1m' + 'Coherence Score = ', temp_coherenace)

    

    df_topic_sents_keywords = format_topics_sentences(covid_df_bm25_filter,ldamodel=optimal_model, corpus=corpus, texts=docs)

    df_dominant_topic = df_topic_sents_keywords.reset_index()

    



    topicPercContrib = df_dominant_topic.Topic_Perc_Contrib.sort_values(ascending=False).tolist()    

    print('\033[1m' + 'Dominant Score Selection')

    topic_contrib_x_idx = range(1, len(topicPercContrib)+1)    

    topic_kn = KneeLocator(topic_contrib_x_idx, topicPercContrib,curve='concave',direction='decreasing',online=True)

    

    #optimal_topic_score,kn_topic = kneeLocator(topic_contrib_x_idx ,topicPercContrib)

    

    

    #optimal_bm25_score = bm25_score[bm25_kn.knee-1]

    optimal_topic_score = topicPercContrib[topic_kn.knee-1]

    

    dominant_topic_filtered_df = df_dominant_topic[df_dominant_topic['Topic_Perc_Contrib']>=optimal_topic_score]

    

    print('\033[1m' + 'Number of Papers Selected after Dominant Topic Scoring = ', dominant_topic_filtered_df.shape[0])    

    

    topNStr = 'Top_'+str(top_N_sentence)+'_Sentence'

        

    title_body_str = dominant_topic_filtered_df.title.fillna('') +' ' + dominant_topic_filtered_df.body.fillna('')

    dominant_topic_filtered_df['title_body_clean'] = title_body_str.apply(clean_text).to_frame()



    sentence_embed_df = dominant_topic_filtered_df.title_body_clean.apply(sentenceEmbed,args=[tcount])

    dominant_topic_filtered_df[['Top_3_Sentence',topNStr,'SentenceScore']] = pd.DataFrame(sentence_embed_df.to_list(), columns=['Top_3_Sentence',topNStr,'SentenceScore'], index=sentence_embed_df.index)    

    dominant_topic_filtered_df = dominant_topic_filtered_df.sort_values(by=['SentenceScore'], ascending=False)      

    

    results = dominant_topic_filtered_df[['paper_id','title',topNStr,'Top_3_Sentence','SentenceScore','publish_time','url','all_text']]    

    

    if(visualization):

        finalfig, finalaxis = plt.subplots(1,3,figsize=(20,5))

        plotKnee(bm25_kn,bm25_x_idx,bm25_score,min(bm25_score),max(bm25_score),'BM25 Doc#','BM25 Score',finalaxis,0)

        plotCoherence(start_num_topics, noOfMaxTopics, step_num_topics,coherence_values,finalaxis,1)

        plotKnee(topic_kn,topic_contrib_x_idx,topicPercContrib,min(topicPercContrib),max(topicPercContrib),'Topic Doc ID','Topic % Contribution',finalaxis,2)

        plt.show()

        print('\033[1m' + '\nTop N Topics Word Cloud' + '\033[0m')

        plotWordCloud(number_of_topics,model_topics)



    print('\033[1m' + 'Final Number of Papers Selected = ', results.shape[0])

    print('\033[1m' + 'Subtask: ' + task9[tcount] + '\n')

    print('\033[1m' + '*********************************Completed*****************************************')

    return results;
def run_task(val):

    tcount = list(dict_task.values()).index(val)

    #taskscount=len(task9)

    # taskscount=1

    # tcount = 0

    visualization=False

    taskResults = pd.DataFrame(columns=[])

    #for tcount in range(taskscount):

    fileName = task9[tcount]

    results = bm25res(tcount, visualization)

        #results.to_csv(fileName + '.csv', index=False)   

        #scatterPlot(results,fileName+'.html','Top_'+str(top_N_sentence)+'_Sentence')

    temp = results

    temp['SubTask'] = fileName

    temp['SubTask'] = temp.SubTask.fillna(fileName)

    taskResults = taskResults.append(temp)

    taskResults =taskResults.reset_index(drop=True)

    final_visualization(taskResults)

    

def run_task_all():

    taskscount=len(task9)

    # taskscount=1

    # tcount = 0

    visualization=False

    taskResults = pd.DataFrame(columns=[])

    for tcount in range(taskscount):

        #fileName = 'task9_subtask_' + str(tcount)

        fileName = task9[tcount]

        results = bm25res(tcount, visualization)

            #results.to_csv(fileName + '.csv', index=False)    

            #scatterPlot(results,fileName+'.html','Top_'+str(top_N_sentence)+'_Sentence')

        temp = results

        temp['SubTask'] = fileName

        temp['SubTask'] = temp.SubTask.fillna(fileName)

        taskResults = taskResults.append(temp)

    taskResults =taskResults.reset_index(drop=True)

    final_visualization(taskResults)



def final_visualization(taskResults):

    topNColumnName = 'Top_'+str(top_N_sentence)+'_Sentence'

    taskResults.fillna('',inplace=True)

    #taskResults.to_csv('all_task_final_output.csv', index=False)

    scatterPlot(taskResults,'Task9_Scatterplot.html',topNColumnName)

    tsneplot(taskResults,topNColumnName)

    taskResults['publish_time'] = taskResults['publish_time'].astype(str)

    generate_html_table(taskResults[['SubTask','title','Top_3_Sentence','publish_time','url']])
#Runs all the sub tasks for the task 9

run_task_all()