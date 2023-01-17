!pip install rank_bm25

!pip install sentence_transformers

!pip install yattag
#Import of Required Libraries/functions

import os

import json

import scipy

import datetime as dt

from sentence_transformers import SentenceTransformer

import pickle

import numpy as np

import re

import string

import nltk

from nltk.corpus import stopwords

from textblob import Word

import pandas as pd

nltk.download('stopwords')

exclude_list = string.digits + string.punctuation

table = str.maketrans(exclude_list, len(exclude_list)*" ")

stop = stopwords.words('english')

english_stopwords = list(set(stop))

import matplotlib.pyplot as plt



from gensim.summarization import summarize

# For Visualization

from IPython.core.display import HTML

import ipywidgets as widgets

from ipywidgets import interact, interactive, Layout

import seaborn as sns

#Define dataPath to download files

dataPath = ("kaggleData")

downloadPath = ("/kaggle/input/CORD-19-research-challenge/")

checkfolder = os.path.isdir(dataPath)



# If folder doesn't exist, then create it.

if not checkfolder:

    os.makedirs(dataPath)



#Reading Data from JSON files

from os import walk

rootFolders = next(os.walk(dataPath))[1]
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



extract_func = lambda x, func: ['|'.join(func(i) for i in x)]

format_authors = lambda x: f"{x['first']} {x['last']}"

format_full_authors = lambda x: f"{x['first']} {''.join(x['middle'])} {x['last']} {x['suffix']}"

format_abstract = lambda x: "{}\n {}".format(x['section'], x['text'])

all_keys = lambda x, key: '|'.join(i[key] for i in x.values())
%%time



# This section converts all the JSON files to a aggregated dataframe

aggregate_dflist = []

for path in ['biorxiv_medrxiv', 'comm_use_subset', 'custom_license', 'noncomm_use_subset']:

    path

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

                #'authors': extract_func(paper['metadata']['authors'], format_authors),

                #'full_authors': extract_func(paper['metadata']['authors'], format_full_authors),

                #'affiliations': extract_func(paper['metadata']['authors'], affiliation_parsing),

                #'emails': extract_key(paper['metadata']['authors'], 'email'),              

                'body': extract_func(paper['body_text'], format_abstract),

                #'body_cite_spans': cite_parsing(paper['body_text'], 'cite_spans'),

                #'body_ref_spans': cite_parsing(paper['body_text'], 'ref_spans'),

                #'bib_titles': all_keys(paper['bib_entries'], 'title'),

                #'ref_captions': all_keys(paper['ref_entries'], 'text'),

                #'back_matter': extract_key(paper['back_matter'], 'text')

            })

            df_list.append(paper_df)

            del paper_df

                        

        if len(df_list) > 0:

            temp_df = pd.concat(df_list)

            #temp_df['dataset'] = path+'_'+subpath

            aggregate_dflist.append(temp_df)

            del temp_df

            

aggregate_df = pd.concat(aggregate_dflist)

%%time

#Extract body length is greater than >1200

aggregate_df= aggregate_df[aggregate_df['body'].apply(len) >=1200]

#Merge from Metadata file URL, Journal, Publish time



metadata_df = pd.read_csv(f'{downloadPath}/metadata.csv',parse_dates = ['publish_time'])

aggregate_df = pd.merge(aggregate_df, metadata_df[['sha','publish_time','url' ]], left_on = 'paper_id', right_on = 'sha', how = 'left')
#Drop the Additional column created

aggregate_df.drop(['sha'], axis = 1, inplace=True)
#Fill the empty columns

aggregate_df.fillna({'publish_time':dt.datetime(2020,1,1), 'url':'UNSPECIFIED'}, inplace=True)
#aggregate_df.drop(['publish_time','url'], axis = 1, inplace=True)

aggregate_df.count()
#Process functions for BM-25



SEARCH_DISPLAY_COLUMNS = ['paper_id','title', 'body', 'url']



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



# BM25 



class WordTokenIndex:

    

    def __init__(self, 

                 corpus: pd.DataFrame, 

                 columns=SEARCH_DISPLAY_COLUMNS):

        self.corpus = corpus

        raw_search_str =self.corpus.title.fillna('') +' ' + self.corpus.body.fillna('')

        #self.corpus['all_text'] = raw_search_str.apply(preprocess).to_frame()

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

        self.bm25 = BM25Okapi(self.index.terms.tolist(),k1=3,b=0.001)

        

    def search(self, search_string, n=4):

        search_terms = preprocess(search_string)

        doc_scores = self.bm25.get_scores(search_terms)

        ind = np.argsort(doc_scores)[::-1][:n]

        results = self.corpus.iloc[ind][self.columns]

        results['Score'] = doc_scores[ind]

        results = results[results.Score > 0]

        return SearchResults(results.reset_index(), self.columns + ['Score'])

    

def show_task(taskTemp,taskId):

    #print(Task)

    keywords = taskTemp#tasks[tasks.Task == Task].Keywords.values[0]

    print(keywords)

    search_results = bm25_index.search(keywords, n=200)    

    return search_results
task9 = ["What has been published about information sharing and inter-sectoral collaboration?",

"What has been published about data standards and nomenclature?",

"What has been published about governmental public health? What do we know about risk communication?",

"What has been published about communicating with high-risk populations?",

"What has been published to clarify community measures?",

"What has been published about equity considerations and problems of inequity?",

"Methods for coordinating data-gathering with standardized nomenclature.",

"Sharing response information among planners, providers, and others.",

"Understanding and mitigating barriers to information-sharing.",

"How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response",

"Integration of federal/state/local public health surveillance systems.",

"Value of investments in baseline public health response infrastructure preparedness",

"Modes of communicating with target high-risk populations (elderly, health care workers).",

"Risk communication and guidelines that are easy to understand and follow.",

"Communication that indicates potential risk of disease to all population groups.",

"Misunderstanding around containment and mitigation.",

"Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity,",

"Funding to ensure all citizens in need are supported and can access information, surveillance, and teatment.",

"Measures to reach marginalized and disadvantaged populations.",

"Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.",

"Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.",

"Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care",

"How is social media used for COVID-19 information sharing and collaboration.",

"How do countries share data on disease spread and effectiveness of government programs",

"How effective has social distancing measures been in countries.",

"What are the popular methods for citizens to get access to COVID-19 information."

]



tasks={'What has been published about information sharing and inter-sectoral collaboration?': task9}
#Load the BM-25 Corpus File



bm25CORPUS_PATH='/kaggle/input/bm25pickle/corpusbm25.pkl'

if not os.path.exists(bm25CORPUS_PATH):

    print("Caching the corpus for future use...")

    with open('corpusbm25.pkl', 'wb') as file:

        pickle.dump(bm25_index, file)

    with open(bm25CORPUS_PATH, 'rb') as corpus_pt:

        bm25corpus = pickle.load(corpus_pt)    

        

else:

    print("Loading the corpus from", bm25CORPUS_PATH, '...')

    with open(bm25CORPUS_PATH, 'rb') as corpus_pt:

        bm25corpus = pickle.load(corpus_pt)
#Create Plot to show no. of articles in corpus related to Task7

import collections

dict1 = {}

for task in task9:

    task_text = ' '.join(preprocess(task)[:8])

    dict1[task_text] = bm25corpus.search(task,n=aggregate_df.shape[0]).getDf().shape[0]

    



dict1_sorted = sorted(dict1.items(), key=lambda kv: kv[1])



fig = plt.figure(figsize=(16,16))



height = [tpl[1] for tpl in dict1_sorted]

bars = [tpl[0] for tpl in dict1_sorted]

#bars = [tpl[0] for tpl in task]

y_pos = np.arange(len(bars))

 

plt.barh(y_pos, height)

plt.yticks(y_pos, bars)



plt.title("No of Articles from BM25 Algorithm for What do we know about diagnostics and surveillance?")

plt.xlabel("# Articles")

plt.ylabel("Tasks") 

    

plt.show()
#Load SciBERT Model

MODEL_PATH='/kaggle/input/scibertnli/scibert-nli'

model = SentenceTransformer(MODEL_PATH)
#Corpus Building and mapping index of Paper-ID

corpus_dictionary = {}

corpus = []

idx = 0

for _, row in aggregate_df.iterrows():

    if isinstance(row['body'], str):

        if row['body'] != "Unknown":

            if len(row['body']) > 900:

                corpus.append(row['body'])

                corpus_dictionary[idx] = row['paper_id']   

                idx +=1
#Building/Loading Corpus

CORPUS_PATH='/kaggle/input/scibertnli/corpus.pkl'

if not os.path.exists(CORPUS_PATH):

    print("Caching the corpus for future use...")

    with open('corpus.pkl', 'wb') as file:

        pickle.dump(corpus, file)

else:

    print("Loading the corpus from", CORPUS_PATH, '...')

    with open(CORPUS_PATH, 'rb') as corpus_pt:

        corpus = pickle.load(corpus_pt)
#Building/Loading Embeddings

EMBEDDINGS_FILE='/kaggle/input/scibertembeddings/scibertnliembeddings.pkl'

if not os.path.exists(EMBEDDINGS_FILE):

        print("Computing and caching model embeddings for future use...")

        embeddings = model.encode(corpus, show_progress_bar=True)

        with open(EMBEDDINGS_FILE, 'wb') as file:

            pickle.dump(embeddings, file)

else:

        print("Loading model embeddings from", EMBEDDINGS_FILE, '...')

        with open(EMBEDDINGS_FILE, 'rb') as file:

            embeddings = pickle.load(file)
#Process Text

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



def strip_characters(text):

    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)

    t = re.sub('/', ' ', t)

    t = t.replace("'",'')

    return t
#Task9 Questions

task9 = [

"What has been published about information sharing and inter-sectoral collaboration?",

"What has been published about data standards and nomenclature?",

"What has been published about governmental public health? What do we know about risk communication?",

"What has been published about communicating with high-risk populations?",

"What has been published to clarify community measures?",

"What has been published about equity considerations and problems of inequity?",

"Methods for coordinating data-gathering with standardized nomenclature.",

"Sharing response information among planners, providers, and others.",

"Understanding and mitigating barriers to information-sharing.",

"How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response",

"Integration of federal/state/local public health surveillance systems.",

"Value of investments in baseline public health response infrastructure preparedness",

"Modes of communicating with target high-risk populations (elderly, health care workers).",

"Risk communication and guidelines that are easy to understand and follow.",

"Communication that indicates potential risk of disease to all population groups.",

"Misunderstanding around containment and mitigation.",

"Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity,",

"Funding to ensure all citizens in need are supported and can access information, surveillance, and teatment.",

"Measures to reach marginalized and disadvantaged populations.",

"Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.",

"Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.",

"Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care",

"How is social media used for COVID-19 information sharing and collaboration.",

"How do countries share data on disease spread and effectiveness of government programs",

"How effective has social distancing measures been in countries.",

"What are the popular methods for citizens to get access to COVID-19 information."

 

]



tasks={'What do we know about diagnostics and surveillance?': task9}

# Function to find out relavent articles using 3 distance measures. 

def ask_question_cosine(query, model, corpus, corpus_embed, task, measure, top_k=5):   

    queries = [query]

    query_embeds = model.encode(queries, show_progress_bar=False)

    for query, query_embed in zip(queries, query_embeds):

        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, measure)[0]

        distances = zip(range(len(distances)), distances)

        distances = sorted(distances, key=lambda x: x[1])

        

        results = []

        

        for count, (idx, distance) in enumerate(distances[:top_k]):

            results.append([count + 1, corpus[idx].strip(), task, measure, corpus_dictionary[idx], round(1 - distance, 4)])

            

    return results
# Function to create results in a dataframe

def get_bert_results(tasks, model, corpus, embeddings, distance_measure):   

    results = []

    

    #distance_measure = ['cosine','chebyshev','canberra']

    #distance_measure = ['cosine']   

    for task in tasks:

        task_text = ' '.join(preprocess(task)[:8])

        for dist_measure in distance_measure:

            results.append(ask_question_cosine(task_text, model,corpus,embeddings,task,dist_measure))

    

    return results





def get_bert_result_df(df, tasks, model, corpus, embeddings):

    bertresults_df = pd.DataFrame(columns=['Task', 'Distance_Measure','URL', 'Publish_Date','Title','Paper_Id','Summary' ])

    #bertresults_df = pd.DataFrame(columns=['Task', 'Distance_Measure','Score','URL', 'Publish_Date', 'Journal', 'Title','paper_id' ])

   

    #bert_results = []

    #article_ids = []

    distance_measure = ['cosine','cityblock','sqeuclidean']

    bert_results= get_bert_results(tasks, model, corpus, embeddings, distance_measure)

    #bert_results= get_bert_results(tasks, model, corpus, embeddings,distance_measure)

    #article_ids = get_article_ids(df, bert_results)

    

    for ber in bert_results:

        for rel in ber:

            matched_row = df[df['paper_id']==rel[4]]

            url = matched_row.url.values[0]

            publish_time = matched_row.publish_time.values[0]

            #journal = matched_row.journal.values[0]

            title = matched_row.title.values[0]

            bertsummary = summarize(rel[1],ratio=0.02)

            #score = str(int(round(rel[5]*100))) 

            #score = str(rel[5])

            task = rel[2]

            dist = rel[3]

            #bertresults_df = bertresults_df.append({'Task':task, 'Distance_Measure':dist,'URL':url, 'Publish_Date':publish_time, 'Journal':journal, 'Title':title,'SciBERTSummary':bertsummary, 'paper_id':rel[4]},ignore_index=True)

            bertresults_df = bertresults_df.append({'Task':task, 'Distance_Measure':dist, 'URL':url, 'Publish_Date':publish_time, 'Title':title,'Paper_Id':rel[4],'Summary':bertsummary},ignore_index=True)

    

    return bertresults_df 
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
%%time

#Run the model and generate results

bert_results_df = get_bert_result_df(aggregate_df, task9, model, corpus, embeddings)
#Drop the duplicate articles extracted by all three distance measures.

bert_results_df.drop_duplicates(subset='Paper_Id', keep="first", inplace=True)

bert_results_df['Publish_Date'] = bert_results_df['Publish_Date'].dt.date
#Provides top 5 Articles

bert_results_df.head(5)
# Save the HTML for later viewing

berthtml = generate_html_table(bert_results_df.sort_values(by=['Publish_Date'], ascending=False))

#display(HTML(berthtml))

%store berthtml >results.html
#Sort the results in decending order by Publish Data

bert_results_df.sort_values(by=['Publish_Date'], ascending=False,inplace=True)
#Interactive way to show the top Journals

task_list = list(np.unique(bert_results_df['Task'].tolist()))

task_list.insert(0, "Please select a question")



cols = ["Paper_Id", "Title", "URL", "Publish_Date", "Summary"]



@interact

def dropdowns(Question = task_list):

    if Question == "Please select a question":

        pass

    else:   

        print (Question)

        display(HTML(generate_html_table(bert_results_df[bert_results_df['Task'].str.strip() == str(Question).strip()][cols])))
#Plot to show Distance Measure Mix

ax = sns.countplot(x="Distance_Measure", data = bert_results_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)