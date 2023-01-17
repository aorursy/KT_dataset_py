## Earliest publication year

earliest_publication_year = 2002 # this is the year where SARS outbreak happened

## Questions

questions = [

    'How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).',

    'Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.',

    'Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and nonprofit, including academic), including legal, ethical, communications, and operational issues.',

    'National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).',

    'Development point of care test (like a rapid influenza test) and rapid bedside tests, recognizing the tradeoffs between speed, accessibility, and accuracy.',

    'Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which  are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).',

    'Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.',

    'Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.',

    'Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.',

    'Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.',

    'Policies and protocols for screening and testing.',

    'Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.',

    'Technology roadmap for diagnostics.',

    'Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.',

    'New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID19 and future diseases.',

    'Coupling genomics and diagnostic testing on a large scale.',

    'Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.',

    'Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturallyoccurring pathogens from intentional.',

    'One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.'

]

## keywords

keywords = [

    'widespread current exposure immediate policy recommendations mitigation measures denominators case fatality rate testing mechanism sharing information demographics Sampling methods asymptomatic carriers transmission disease serosurveys convalescent samples early detection disease screening neutralizing antibodies ELISA IgG enzymelinked immunosorbent assay',

    'efforts increase capacity existing diagnostic platforms existing surveillance platforms National Respiratory Enteric Virus Surveillance System (NREVSS)', 

    'Recruitment support coordination local expertise public private commercial nonprofit NPO academic legal ethical communications operational issues',

    'USA U.S. National Federal guidance guidelines best practices standards states Communications leverage universities private laboratories testing purposes communications public health officials public)',

    'care rapid influenza test speed accuracy accessibility Development care test Detection pandemic influenza accuracy rapid influenza testing Diagnostic performance nearpatient testing influenza',

    'pattern pcr intention invention innovation conception performance implementation surveillance potential testers examination composition entity aid longitudinal vital decisive interpret translate tool distributed disease surveillance Rapid design execution targeted surveillance potential testers PCR COVID understanding impact  adhoc local interventions COVID Social Contact Network Epidemic Outbreak',

    'interval assay evolution growth maturation ontogeny exploitation developing sector sphere aid quickly rapidly migrate daignostic assays migrate assays devices private sector infectious diseasea assay development issues',

    'development evolution phylogeny virus tocopherol familial genetic hereditary inherited transmitted transmissible drift impetus trend surveillanced etection sensing detecting Efforts track evolution virus evolution virus genetic drift influenza virus mutation covid usa tracking pandemic virus computational analysis mutation',

    'Latency issues Latency period Detect pathogen Viral load Biological sampling Environmental sampling Proactive holistic approach Detecting viruses early source Zoonotic diseases Risk exposure people Surveillance science Diagnostic technologies early detection identification Contracting disease displaying symptoms Time lag testing patients Incubation period Surface samples',

    'Diagnostics Host response markers Detect early disease Predict severe disease progression Understanding best clinical practice Therapeutic interventions Therapeutic Management Treatment guidelines Symptom screening',

    'Policies Protocols screening testing Interstate travel Global travel increased testing capacity Social distancing Selfquarantine rules Contact tracing',

    'Policies Mitigate effects supplies Mass testing Swabs reagents Efficient emergency Management system National stockpile Wartime presidential power Task force',

    'geospatial data visualization platform technology Tectonix cellphone heat maps detecting movement virus hotspots mobile phone heat map technology detecting virus hotspot movement sentinel fever tracker technology technology infrastructure Rapid Sharing covid19 Results technology platform Rapid Sharing covid19 Results ventilator sharing technology convid19 surveillance technology mass screening infrared thermal detection technology systems remote health monitoring technology covid19 SARSCoV2 RNA signature detection technology covid19 SARSCoV2 detection technology loopmediated isothermal amplification LAMP covid19 SARSCoV2 remote patient monitoring technology covid19 SARSCoV2 mass antibody testing platform early detection of lung infiltrate',

    'Developing Scaling up Diagnostic tests Future coalition Accelerator models Streamlined regulatory environment Detect National, Global collaboration Global Health Security Translational medicine Novel therapeutics Automation Technological innovation FDA Laboratory Accurate, Reliable, Reproducible Testing Global access Medical countermeasures National policy framework WHO Legal, regulatory, logistical and funding barriers the highest bidder?',

    'New platforms Technology Improve response time holistic approach Future diseases Pathogenesis Immunity Transdisciplinary approaches Transboundary health issues Social–ecological understanding Diagnosis RNA genome Microbial infection Serological assay Respiratory infection Genomic characteristics Molecular tests Rapid detection Early identification Screening patients Virus surveillance Detection Assays',

    'genetic synthesis Gene Sequencing Assays detect SARSCoV2 genetic code genome sequencing RNA Genome sequence DNA genes sequencing Nucleic acidbased Nucleic acid genomic sequences diagnosis clinical testing testing Detecting, detect, detection diagnostic assays Diagnose diagnostics Largescale Large scale',

    'rapid sequencing bioinformatics target regions of the genome variant covid19 SARS SARSCoV2 ebola',

    'technology data sequencing advanced analytics unknown pathogens capabilities distinguishing naturallyoccurring pathogens intentional Covid19; SARS SARSCov ebola',

    'health surveillance virus spillover animals humans future pathogens evolutionary hosts bats transmission hosts heavily trafficked farmed wildlife domestic food companion species environmental demographic Covid19 SARS Cov2 SARS ebola'

    ]
!pip install rank_bm25

!pip install swifter

!pip install yattag

!pip install langdetect
#All the import functions



#Import required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import json

import os

from matplotlib import colors

# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models.coherencemodel import CoherenceModel

from gensim.models.ldamodel import LdaModel

from gensim.corpora.dictionary import Dictionary



# spacy for lemmatization

import spacy



from rank_bm25 import BM25Okapi

from tqdm import tqdm

import nltk

from nltk.tokenize import word_tokenize, sent_tokenize , RegexpTokenizer

from nltk.stem import WordNetLemmatizer, PorterStemmer

from nltk.corpus import wordnet 

from nltk.corpus import stopwords

from difflib import SequenceMatcher , get_close_matches, Differ

from wordcloud import WordCloud, STOPWORDS

import matplotlib.colors as mcolors

import string

from textblob import Word

import collections

import scattertext as st

import prettytable

import zipfile

import textwrap

import datetime as dt

import swifter

import multiprocessing as mp

import gc





#nltk.download()

nltk.download('words')

nltk.download('stopwords')

nltk.download('punkt')





stop_words = set(stopwords.words())

nltkwords = set(nltk.corpus.words.words())





from tqdm._tqdm_notebook import tqdm_notebook

tqdm_notebook.pandas()



from IPython.display import IFrame

import glob

from nltk.tokenize.toktok import ToktokTokenizer

from yattag import Doc, indent

from langdetect import detect



#for topic modeling

from sklearn.decomposition import LatentDirichletAllocation as LDA

from sklearn.feature_extraction.text import CountVectorizer

from pyLDAvis import sklearn as sklearn_lda

import pyLDAvis

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import ipywidgets as widgets

from ipywidgets import interact, interactive, Layout

import warnings

warnings.filterwarnings('ignore') # warnings are not needed in production code
downloadPath = ('/kaggle/input/CORD-19-research-challenge')
#Define utilities for data parsing



def affiliation_parsing(x: dict) -> str:

    """Parse affiliation into string."""

    current = []

    for key in ['laboratory', 'institution']:

        if x['affiliation'].get(key):  # could also use try, except

            current.append(x['affiliation'][key])

        else:

            current.append('')

    for key in ['addrLine', 'settlement', 'region', 'country', 'postCode']:

        if x['affiliation'].get('location'):

            if x['affiliation']['location'].get(key):

                current.append(x['affiliation']['location'][key])

        else:

            current.append('')

    return ', '.join(current)





def cite_parsing(x: list, key: str) -> list:

    """Parse references into lists for delimiting."""

    cites = [i[key] if i else '' for i in x] #test['body_text']]

    output = []

    for i in cites:

        if i:

            output.append(','.join([j['ref_id'] if j['ref_id'] else '' for j in i]))

        else:

            output.append('')

    return '|'.join(output)





def extract_key(x: list, key:str) -> str:

    if x:

        return ['|'.join(i[key] if i[key] else '' for i in x)]

    return ''



# This function is used to return the year of the latest cited paper

def latest_cited_paper_year(x):

    try: 

        years = [i['year'] for i in x.values() if (i['year'] != None)]

        today = dt.datetime.today()

        today_year = today.year

        # cap years to today's year

        years = [year for year in years if year <= today_year]

        if (years != None) & (years != []):

            return dt.datetime(max(years),1,1)

        else:

            return None

    except:

        return None



extract_func = lambda x, func: ['|'.join(func(i) for i in x)]

format_authors = lambda x: f"{x['first']} {x['last']}"

format_full_authors = lambda x: f"{x['first']} {''.join(x['middle'])} {x['last']} {x['suffix']}"

format_abstract = lambda x: "{}\n {}".format(x['section'], x['text'])

all_keys = lambda x, key: '|'.join(i[key] for i in x.values())
%%time



# This section converts all the JSON files to a aggregated dataframe



aggregate_dflist = []

for path in ['biorxiv_medrxiv', 'comm_use_subset', 'custom_license', 'noncomm_use_subset']:

    for subpath in ['pdf_json','pmc_json']:

        if not os.path.exists(f'{downloadPath}/{path}/{path}/{subpath}/') : continue

        json_files = [file for file in os.listdir(f'{downloadPath}/{path}/{path}/{subpath}/') if file.endswith('.json')]

        df_list = []

                

        for js in json_files:

            

            with open(os.path.join(f'{downloadPath}/{path}/{path}/{subpath}', js)) as json_file:

                paper = json.load(json_file)

                

            paper_df = pd.DataFrame({

                'paper_id': paper['paper_id'],

                'title': paper['metadata']['title'],

                'authors': extract_func(paper['metadata']['authors'], format_authors),

                'full_authors': extract_func(paper['metadata']['authors'], format_full_authors),

                'affiliations': extract_func(paper['metadata']['authors'], affiliation_parsing),

                'emails': extract_key(paper['metadata']['authors'], 'email'),              

                'body': extract_func(paper['body_text'], format_abstract),

                'body_cite_spans': cite_parsing(paper['body_text'], 'cite_spans'),

                'body_ref_spans': cite_parsing(paper['body_text'], 'ref_spans'),

                'bib_titles': all_keys(paper['bib_entries'], 'title'),

                'latest_cited_paper_year': latest_cited_paper_year(paper['bib_entries']),

                'ref_captions': all_keys(paper['ref_entries'], 'text'),

                'back_matter': extract_key(paper['back_matter'], 'text')

            })

            df_list.append(paper_df)

            

        if len(df_list) > 0:

            temp_df = pd.concat(df_list)

            temp_df['dataset'] = path+'_'+subpath

            aggregate_dflist.append(temp_df)

aggregate_df = pd.concat(aggregate_dflist)

del temp_df 

del aggregate_dflist
######################## Remove duplicate articles ##############

aggregate_df = aggregate_df[(~aggregate_df['title'].duplicated()) | (aggregate_df['title']=='')]
#Merge from Metadata file URL, Journal, Publish time

metadata_df = pd.read_csv(f'{downloadPath}/metadata.csv',parse_dates = ['publish_time'])

covid_csv = pd.merge(aggregate_df, metadata_df[['sha','journal','publish_time','url' ]], left_on = 'paper_id', right_on = 'sha', how = 'left')

covid_csv.drop(['sha','bib_titles','ref_captions','body_cite_spans','back_matter','emails','body_ref_spans'], axis = 1, inplace=True)

del metadata_df

del aggregate_df

gc.collect();
## Remove articles that has text less than 1200 characters

covid_csv = covid_csv[covid_csv['body'].str.len() > 1200]
# Fill the missing publish times with approximate time using the year of the latest cited article in the paper

print('Percentage of articles with missing publish-time BEFORE filling missing values = ' +

       str(round(100*sum(covid_csv['publish_time'].isna())/len(covid_csv))) + '%'

     )



covid_csv['publish_time'].fillna(covid_csv['latest_cited_paper_year'], inplace=True)

covid_csv.drop(['latest_cited_paper_year'], axis = 1, inplace=True)



print('Percentage of articles with missing publish-time AFTER filling missing values = ' +

       str(round(100*sum(covid_csv['publish_time'].isna())/len(covid_csv))) + '%'

     )

# fill the remaining publish_time values with 1900,7,1

covid_csv.fillna({'publish_time':dt.datetime(1900,7,1), 'url':'UNSPECIFIED'}, inplace=True)
########### Gets dataset after a particular earliest_publication_year, if earliest_publication_year = -1 gets whole dataset #########

#Functions

def get_year(x):

    year = int(x.split('-')[0])

    return year



def get_year_timestamp(x):

    year = x.year

    return year





def get_df_afterdate(df, date):

    if date == -1:

        df2 = df

    else:

        df2 = df[df['publish_time'].apply(get_year_timestamp) >= date]

    return df2    



print('Number of total articles BEFORE filtering the articles by earliest_publication_year= ' +

      str(len(covid_csv)))

covid_csv = get_df_afterdate(covid_csv, earliest_publication_year)

print('Number of total articles After filtering the articles by earliest_publication_year= ' +

      str(len(covid_csv)))
##### Keep only english papers

def detect_lang(text):

    try:

        portion=text[1000:2000]

        lang=detect(portion)

    except Exception:

        lang=None

    

    return lang

covid_csv['Language'] = covid_csv['body'].progress_apply(detect_lang)

covid_csv = covid_csv[covid_csv['Language'] == 'en']

del covid_csv['Language']

gc.collect();
#this function plot the histogram of dates of articles 

def get_date_hist(df):

    plt.rcParams['figure.figsize'] = (15,6)

    fig, ax = plt.subplots(1,1) 

    

    years = df['publish_time'].apply(get_year_timestamp).values

    years, counts = np.unique(years, return_counts=True)

    ax.bar(years, counts)

    ax.set_xticks(years)

    ax.set_xticklabels(years)

    ax.set_title('Histogram of date of articles ')

    ax.set_xlabel('Date of article')

    ax.set_ylabel('Number of articles')



    

    plt.show()

    

get_date_hist(covid_csv)
%%time

####### Changing words to lower case + Remove stop_words +Lemmatization + Stemming ##########

def getCleanText(text):

    if type(text) != str or len(text) <1000 :

        return ''

    toktok = ToktokTokenizer()

    tokens = toktok.tokenize(text)

    # convert to lower case

    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word

    table = str.maketrans('', '', string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic

    words = [word for word in stripped if word.isalpha() and word not in stop_words and word in nltkwords]

    text = " ".join(words)

    return text

#preprocessing 'body' attribute

covid_csv['processed_body'] = covid_csv.progress_apply(lambda row: getCleanText(row['body']), axis=1)
# decide whether to use questions or keywords for answer retrival based on 

# user input

try:

    keywords

except NameError:

    # keywords is not defined, use questions for retriving the answers

    queries = questions

else:

    if len(questions) == len(keywords):

        # keywords is defined and it has the same length as questions, 

        # use keywords for retriving the answers

        queries = keywords

    else:

        # keywords is defined but it does not have the same length as questions, 

        # use questions for retriving the answers

        queries = questions



#Convert queries to lower case

queries = [query.lower() for query in queries]
def get_top100_bm25_per_question(question_id, questions, bm25_index, original_df_corpus, orignal_df):

    #tokenize queries          

    tokenized_query = questions[question_id].split(" ")

   

    top_doc_list = bm25_index.get_top_n(tokenized_query, original_df_corpus, n=100)   

    top_doc_list_paper_id = [orignal_df.iloc[original_df_corpus.index(top_doc_list[j])].paper_id for j in range(len(top_doc_list))]

    

    return top_doc_list_paper_id

   
def get_top100_bm25_questions(questions, orignal_df):

    corpus = [orignal_df['processed_body'].iloc[i] for i in range(len(orignal_df))]

    tokenized_corpus = [gensim.utils.simple_preprocess(orignal_df['processed_body'].iloc[i]) for i in range(len(orignal_df))]

    

    print('Generating BM25 index...')

    bm25_index = BM25Okapi(tokenized_corpus,k1=11.5,b=0.001)

    print('BM25 Index generated')

    

    top_docs_df = pd.DataFrame(columns=['question', 'top_docs'])

    for i in range(len(questions)):

        top_doc_list_paper_id = get_top100_bm25_per_question(i, questions, bm25_index, corpus, orignal_df)

        top_docs_df = top_docs_df.append({'question': questions[i], 'top_docs': top_doc_list_paper_id}, ignore_index=True) 

    

    return top_docs_df   
def read_corpus2(df, tokens_only=False):

    for i in range(len(df)):

        tokens = gensim.utils.simple_preprocess(df['processed_body'].iloc[i])      

        if tokens_only:

            yield tokens

        else:

            yield gensim.models.doc2vec.TaggedDocument(tokens, [df['paper_id'].iloc[i]])

#save topn documents returned from BM25 in a dataframe (paper_id, processed_text)

def get_topn_dataframe(listOfIds, origin_df):

#     top_ar = literal_eval(listOfIds)

    top_ar = listOfIds

    text = []

    for j in top_ar:

        row_index = origin_df.index[origin_df.paper_id == j]

        text.append(origin_df.loc[row_index]['processed_body'].values[0])



    top_docs_out = pd.DataFrame(columns=['paper_id', 'processed_body'])

    top_docs_out['paper_id'] = top_ar

    top_docs_out['processed_body'] = text

    

    return top_docs_out
#get most similar documents to the question using doc2vec

def get_top10_doc2vec_per_question(question_id, questions, top100, original_df):

    top_docs = get_topn_dataframe(top100.top_docs[question_id], original_df)

    top_docs = top_docs.append({'paper_id': str(question_id), 'processed_body': questions[question_id]}, ignore_index=True)



    #get training set and build doc2vec model

    train_corpus = list(read_corpus2(top_docs))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=120, 

#                                           min_count=10, 

                                          dm_mean=1, 

                                          epochs=100,

                                          window = 2,

#                                           batch_words=25000, 

                                          workers=mp.cpu_count(), 

                                          seed=420)

    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # Free up some memory

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True) 

    

    question_corpus_id = 100

    inferred_vector = model.infer_vector(train_corpus[question_corpus_id].words)

    sims = model.docvecs.most_similar([inferred_vector], topn=10)

    top_ids = [sim[0] for sim in sims]

    

    return top_ids

#get most similar documents to all questions using doc2vec

def get_top10_doc2vec_questions(questions, top100, original_df):

    top_docs2 = pd.DataFrame(columns=['question', 'top_docs'])

    

    for i in range(len(top100.top_docs)):

#     for i in range(0,1):

        top10_ids = get_top10_doc2vec_per_question(i, queries, top100, original_df)

        top_docs2 = top_docs2.append({'question': top100.question[i], 'top_docs': top10_ids}, ignore_index=True)

    

    return top_docs2
%%time

top100 = get_top100_bm25_questions(queries, covid_csv)
%%time

top10 = get_top10_doc2vec_questions(queries, top100, covid_csv)
#add html text to visualization output

def add_html_text(file_name, q_id, text):

    f = open(file_name,'a+')



    message = """<html>

    <head></head>

    <body><b>Question </b><b>""" + str(q_id+1) +""":</b>"""+ text+"""</body>

    </html>"""



    f.write(message)

    

    f.close()

    

#generate topic modeling visualization for one question

def topic_modeling_visualization(question_id_to_be_visualized, top10, questions, original_df): 

    top10_qi = top10.top_docs.iloc[question_id_to_be_visualized]

    if str(question_id_to_be_visualized) in top10_qi:

        top10_qi.remove(str(question_id_to_be_visualized))

    corpus = [original_df.processed_body[(original_df.paper_id == top10_qi[i])].values[0] for i in range(len(top10_qi))]

    count_vectorizer = CountVectorizer()

    count_data= count_vectorizer.fit_transform(corpus)



    # Create and fit the LDA model

    number_topics = 4

    lda = LDA(n_components=number_topics,random_state=0)

    lda.fit(count_data)

    

    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)

    file_html = 'Topic_Modeling_Visualization/topic_modeling_visualization_for_Q_'+str(question_id_to_be_visualized+1)+'.html'  



    pyLDAvis.save_html(LDAvis_prepared, file_html)

    add_html_text(file_html, question_id_to_be_visualized, questions[question_id_to_be_visualized] )

    



#generate topic modeling visualization for all questions 

def topic_modeling_visualization_all(top10, questions, original_df):

    if not os.path.exists('Topic_Modeling_Visualization'):

        os.makedirs('Topic_Modeling_Visualization')



    for question_id_to_be_visualized in range(len(questions)):

        topic_modeling_visualization(question_id_to_be_visualized, top10, questions, original_df)



#generate topic modeling visualization for all questions    

topic_modeling_visualization_all(top10, questions, covid_csv)   
def read_corpus3(df, tokens_only=False):

    for i in range(len(df)):



        tokens = gensim.utils.simple_preprocess(df['sentences'].iloc[i])



        if tokens_only:

            yield tokens

        else:



            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

# split each article into sentences, insert the question for that article and then make a dataframe

def split_article(article, question=None, insert_question = True):



    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    split_text = nltk_tokenizer.tokenize(article)



    if insert_question == True:

        split_text.insert(0, question)



    article_df =  pd.DataFrame(columns=['sentences'])

    article_df['sentences'] = split_text

    return article_df
%%time

answers_df = pd.DataFrame(columns=['Task Query', 'URL', 'publish_date', 'paper_id', 'Title', 'Highlights'])







for num_q, question in enumerate(top10.question):

    answers_per_ques = []  #list of all answers from documents for one question

    titles  = []  #list of all titles of top articles





    top_docs_list = top10.top_docs.loc[num_q]  #ids from one question

    for num_d, doc in enumerate(top_docs_list):

        if not doc in [str(x) for x in range(len(questions))]:



            article = str(covid_csv.body.loc[covid_csv.paper_id == doc].values)

            title = str(covid_csv.title.loc[covid_csv.paper_id == doc].values)

            url = str(covid_csv.url.loc[covid_csv.paper_id == doc].values).strip("'[]'")

            publish_date = str(covid_csv.publish_time.loc[covid_csv.paper_id == doc].values)

            if (len(publish_date) >= 12):

                    publish_date = publish_date[2:12] # extract just the date

            

            article_df = split_article(article, question)



            train_corpus = list(read_corpus3(article_df))

            model = gensim.models.doc2vec.Doc2Vec(vector_size=100, 

                                                  min_count=1,

                                                  dm_mean=1,

                                                  epochs=150,

                                                  workers=mp.cpu_count(),

                                                  seed=430)

            model.build_vocab(train_corpus)

            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

            # Free up some memory

            model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True) 

            

            inferred_vector = model.infer_vector(train_corpus[0].words,epochs=300)

            sims = model.docvecs.most_similar([inferred_vector], topn=5)  #top 5 sentences

            top_ids = [sim[0] for sim in sims if sim[0]!=0]

            top_sents = article_df.iloc[top_ids]



            answer = ' '.join(top_sents.sort_index().sentences)

            answers_per_ques.append(answer)

            titles.append(title) 

            query = questions[num_q]

            answers_df = answers_df.append({'Task Query':query, 'URL':url ,'publish_date':publish_date ,'paper_id':doc, 'Title':title, 'Highlights':answer}, ignore_index = True)





# HTML

# This function block helps to create a nice HTML format to visualize

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



    from yattag import Doc, indent

    doc, tag, text, line = Doc().ttl()



    with tag("head"):

        with tag("style"):

            text(css_style)



    with tag('table', klass='paleBlueRows'):

        with tag("tr"):

            for col in list(df.columns):

                with tag("th"):

                     with tag("div", klass = "scrollable"):

                        text(col)

        for idx, row in df.iterrows():

            with tag('tr'):

                for i in range(len(row)):

                    with tag('td'):

                        with tag("div", klass = "scrollable"):

                            try:

                                if "http" in str(row[i]):

                                    with tag("a", href = str(row[i])):

                                        text(str(row[i]))

                                else:

                                    text(str(row[i]))

                            except:

                                print(row[i])



    #display(HTML(doc.getvalue()))

    return(doc.getvalue())
# Save in HTML

answers_html = generate_html_table(answers_df)

%store answers_html >Task7_Answers.html

# Save in Excel

answers_df.to_excel("Task7_Answers.xlsx", freeze_panes=(1,1)) 
%%time

# Display only the top paper

from IPython.display import Markdown, display

def printmd(string):

    display(Markdown(string))

    

for num_q, question in enumerate(questions):

    

    printmd(f'**|Question #{num_q+1}|**')

    printmd(question)

    print('='*70)

    

    #find rows in answers_df related to this question

    sliced_answers = answers_df.loc[answers_df['Task Query']==question].reset_index()

    sliced_answers.drop(sliced_answers.tail(len(sliced_answers)-1).index,inplace=True) # Keep only the top paper to simplify the plotting



    for i, highlight in enumerate(sliced_answers.Highlights):

            

            #Title

            title = sliced_answers.Title[i]

            if len(title) > 10:

                title = title[2:-2]

            

            publish_date = sliced_answers.publish_date[i]



            #URL

            url = sliced_answers.URL[i]



            print('Title: ' + title)

            print('Publication_date: ' + publish_date)

            print('URL: ' + url)

            print('Highlights: ' + highlight)

    print('='*70)
from IPython.display import display

def plot_word_cloud_given_text(text,title):

    # Create and generate a word cloud image:

    wordcloud = WordCloud(background_color='black').generate(text)

    # Display the generated image:

    fig = plt.figure(figsize=[10,10])

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    printmd(title)

    plt.show()

    return fig

    #print(title)

def plot_word_cloud_given_question(question,title):

    #find rows in answers_df related to this question

    sliced_answers = answers_df.loc[answers_df['Task Query']==question].reset_index()

    paper_text = ''

    for highlight in sliced_answers.Highlights:

        paper_text += highlight

    if (len(paper_text) == 0):

        print(sliced_answers)

    plot_word_cloud_given_text(paper_text,title)
for num_q, question in enumerate(questions):

    plot_word_cloud_given_question(question, f'Question #{num_q+1}: ' + question)

    printmd('='*70)