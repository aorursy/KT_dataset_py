# Loading all packages 
import os
import json
from pprint import pprint
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow_text
import glob
import matplotlib.pyplot as plt
import scipy as sc
import warnings
import faiss  
import requests
import pickle
from sklearn.metrics.pairwise import cosine_similarity
plt.style.use('ggplot')
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
from flask import Flask
import os
import requests
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
import dash_bootstrap_components as dbc
## Helper Functions
def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)

def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        if 'abstract' in file:
            features = [
                file['paper_id'],
                file['metadata']['title'],
                format_authors(file['metadata']['authors']),
                format_authors(file['metadata']['authors'], 
                               with_affiliation=True),
                format_body(file['abstract']),
                format_body(file['body_text']),
                format_bib(file['bib_entries']),
                file['metadata']['authors'],
                file['bib_entries']
            ]
        else:
            features = [
                file['paper_id'],
                file['metadata']['title'],
                format_authors(file['metadata']['authors']),
                format_authors(file['metadata']['authors'], 
                               with_affiliation=True),
                format_body(file['body_text']),
                format_body(file['body_text']),
                format_bib(file['bib_entries']),
                file['metadata']['authors'],
                file['bib_entries']
            
            ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df


biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))

all_files = []

for filename in filenames:
    filename = biorxiv_dir + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)

### Biorxiv: Generate CSV

cleaned_files = []

for file in tqdm(all_files):
    features = [
        file['paper_id'],
        file['metadata']['title'],
        format_authors(file['metadata']['authors']),
        format_authors(file['metadata']['authors'], 
                       with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),
        format_bib(file['bib_entries']),
        file['metadata']['authors'],
        file['bib_entries']
    ]
    
    cleaned_files.append(features)

col_names = [
    'paper_id', 
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]

clean_df = pd.DataFrame(cleaned_files, columns=col_names)
clean_df.head()
#Reading all CSV files and Concatenating final result
pmc_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'
pmc_files = load_files(pmc_dir)
pmc_df = generate_clean_df(pmc_files)
# pmc_df.to_csv('clean_pmc.csv', index=False)


pmc_dir_1 = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pmc_json/'
pmc_files_1 = load_files(pmc_dir_1)

pmc_df_1 = generate_clean_df(pmc_files_1)
# pmc_df.to_csv('clean_pmc.csv', index=False)


comm_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'
comm_files = load_files(comm_dir)
comm_df = generate_clean_df(comm_files)



comm_dir_1 = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json/'
comm_files_1 = load_files(comm_dir_1)
comm_df_1 = generate_clean_df(comm_files_1)



noncomm_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'
noncomm_files = load_files(noncomm_dir)
noncomm_df = generate_clean_df(noncomm_files)



noncomm_dir_1 = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json/'
noncomm_files_1 = load_files(noncomm_dir_1)
noncomm_df_1 = generate_clean_df(noncomm_files_1)



df_covid_new = pd.concat([clean_df,pmc_df,pmc_df_1,comm_df,comm_df_1,noncomm_df,noncomm_df_1],axis=0,ignore_index=True)

#Reading Metadata file
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})

df_covid_old = pd.merge(df_covid_new,meta_df[['sha','url']],left_on='paper_id',right_on='sha',how='left')

## Saving the Doc information with their URLs.
df_covid_new[['paper_id','title','url']].to_csv('/kaggle/output/df_docid_with_url.csv')

#Reading all CSV files and Concatenating final result
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
## Creating a dictionary with Document as Key and Paragraphs as text
short_paragraph=[]
dict1 = {}
for i in range(len(df_covid_old)):
#     if dict1[df_covid.loc[i,'paper_id']] is not null:
    dict1[df_covid_old.loc[i,'paper_id']] = re.split(r'(?:\r?\n){1,}', df_covid_old.loc[i,'text'])
## Create Vector Embedding for all the Text Documents and storing in a dictionary with key as docId and values as paragraph embeddings
dict_vector_old = {}
for key in list(dict1.keys()):
    try:
        dict_vector_old[key] = embed(dict1[key])
        print(len(dict_vector_old))
    except:
        continue
## Matching Vector and Text Documents (if embedding vector generation fails, we are ignoring the document)        
dict1_old= {}
for key in list(dict_vector_old.keys()):
    if key in dict1:
        dict1_old[key]=dict1[key]
    print(len(dict1_old))
## Storing the paragraph embeddings as pickle file for further use at /kaggle/output/
with open('/kaggle/output/dict1_text_v6.pickle', 'wb') as handle:
    pickle.dump(dict1_old, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/kaggle/output/dict_vector_v6.pickle', 'wb') as handle:
    pickle.dump(dict_vector_old, handle, protocol=pickle.HIGHEST_PROTOCOL)
df_covid = df_covid_old

## Reading the paragraph embeddings pickle files
with open('/kaggle/output/dict1_text_v6.pickle', 'rb') as handle:
    dict1_text_v1 = pickle.load(handle)

with open('/kaggle/output/dict_vector_v6.pickle', 'rb') as handle:
    dict_vector = pickle.load(handle)

## Building the index for semantic search for faiss
index = faiss.IndexFlatL2(512)   # build the index
for vector in list(dict_vector.keys()):
    index.add(dict_vector[vector].numpy())                  # add vectors to the index

## Creating a list of documents with their docid and paragraph text to get the results
text=[]
docId=[]
for key in list(dict1_text_v1.keys()):

    text.extend(dict1_text_v1[key])
    doc=[key]*len(dict1_text_v1[key])
    docId.extend(doc)
    
## Enter the required query to be searched upon
## Below is the query to search "Seasonality of transmission of corona virus"
search = [''' Seasonality of transmission of corona virus''']

## Creating the embedding vectors for the query
predictions = embed(search)

k = 10                         # we want to see 10 nearest neighbors
D, I = index.search(np.array(predictions,dtype='float32'), k) # sanity check
print(I)
print(D)

for i in range(k):
    print(text[I[0][i]])
    print(docId[I[0][i]])
    print('\n')
    
# Supporting function for calculation of Text summarization using Extraction-based text summarization
external_stylesheets=[dbc.themes.BOOTSTRAP]

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash(name = __name__, server = server, external_stylesheets = external_stylesheets)
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app.config['suppress_callback_exceptions'] = True

df = pd.read_csv('https://raw.githubusercontent.com/rahulpoddar/dash-deploy-exp/master/TASK1_annotated_1_v3.csv', encoding='latin1')

tasks = df['Task Name'].unique().tolist()

def data_prep(inpt): 
    clean_data = []
    article3 = ' '.join(inpt)
    result=re.sub("\d+\.", " ", article3)
    clean_data.append(result)
            
    clean_data = pd.DataFrame(clean_data)
    clean_data.columns = ['Remediation']
    clean_data['Remediation'] = clean_data['Remediation'].astype('str')

    clean_data1 = clean_data['Remediation']
    clean_data2 = []
    regex = r"(?<!\d)[-,_;:()](?!\d)"
    for i in range(1):
        result2 = re.sub(regex,'',clean_data1.loc[i])
        clean_data2.append(result2)
    clean_data2 = pd.DataFrame(clean_data2)
    clean_data2.columns = ['Remediation']
    clean_data2['Remediation'] = clean_data2['Remediation'].astype('str')
    
    return (clean_data2)

def _create_dictionary_table(text_string) -> dict:
   
    # Removing stop words
    stop_words = set(stopwords.words("english"))
        
    words = word_tokenize(text_string)
    
    # Reducing words to their root form
    stem = PorterStemmer()
    
    # Creating dictionary for the word frequency table
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table

def _calculate_sentence_scores(sentences, frequency_table) -> dict:   

    # Algorithm for scoring a sentence by its words
    sentence_weight = dict()

    for sentence in sentences:
        sentence_wordcount = (len(word_tokenize(sentence)))
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence[:7] in sentence_weight:
                    sentence_weight[sentence[:7]] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence[:7]] = frequency_table[word_weight]

        sentence_weight[sentence[:7]] = sentence_weight[sentence[:7]] /(sentence_wordcount_without_stop_words)
      
    return sentence_weight

def _calculate_average_score(sentence_weight) -> int:
   
    # Calculating the average score for the sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]

    # Getting sentence average value from source text
    average_score = (sum_values / (len(sentence_weight)))

    return average_score

def _get_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''

    for sentence in sentences:
        if sentence[:7] in sentence_weight and sentence_weight[sentence[:7]] >= (threshold):
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary

def _run_article_summary(article):
    
    #creating a dictionary for the word frequency table
    frequency_table = _create_dictionary_table(article)

    #tokenizing the sentences
    sentences = sent_tokenize(article)

    #algorithm for scoring a sentence by its words
    sentence_scores = _calculate_sentence_scores(sentences, frequency_table)

    #getting the threshold
    threshold = _calculate_average_score(sentence_scores)

    #producing the summary
    article_summary = _get_article_summary(sentences, sentence_scores, 1 * threshold)

    return article_summary

def _output(inpt):
    new = []
    df = data_prep(inpt)
    df_rem = df['Remediation']
    #sentences = sent_tokenize(df_rem[0])
    summary_results = _run_article_summary(df_rem[0])
    new.append(summary_results)
    return(new)
'''
def generate_summary(task):
    return 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'
'''
def generate_table(dff):
    rows = []
    for i in range(len(dff)):
        row = []
        for col in ['Title', 'Output']:
            value = dff.iloc[i][col]
            url = dff.iloc[i]['URL']
            if col == 'Title':
                cell = html.Td(html.A(href=url, children = value))
            else:
                cell = html.Td(children = value)
            row.append(cell)
        rows.append(html.Tr(row))
    return dbc.Table(
        # Header
        [html.Tr([html.Th(col,  style={'text-align':'center'}) for col in ['Title', 'Search Output']]) ] +
        # Body
        rows,
        bordered=True,
        dark=False,
        hover=True,
        responsive=True,
        striped=True,
    )


app.layout = html.Div([
        html.Div([
        html.H1('COVID-19 Open Research Dataset Challenge (CORD-19)', style = {'margin-left': '10%', 'margin-top': '5%'}),
        html.Hr(),
        html.Div([
        html.H3('Type a general query (e.g. "What is Corona Virus?"):'),
        html.Br(),
        dbc.Input(id = 'general-search', type = 'text', placeholder = 'Type a query', value = ''),
        html.Br(),
        dbc.Button(id='submit-button-state', n_clicks=0, children='Submit', color = "primary", className="mr-2", style = {'margin-left': '46%'}),
        ], style = {'width': '80%', 'margin': 'auto'}),
        html.Hr(),
        html.Div([html.H3('OR')],style = {'margin-left': '48%'}),
        html.Hr(),
        html.Div([
        html.H3('Select a task:'),
        dcc.Dropdown(
        id='task-dropdown',
        options=[
            {'label': i, 'value': i} for i in tasks 
        ],
        placeholder="Select a task",
    ),
        html.Br(),
        html.Div([
                html.H3('Select a sub-task:'),
                dcc.Dropdown(
                        id='sub-task-dropdown',
                        placeholder = "Select a sub-task",
                        ),
                ], id = 'sub-task'),
                ], style = {'margin-left': '10%','margin-right': '10%'}),
    ]),
    html.Hr(),
    html.Div([
                html.H3('Response Summary'),
                html.Div(id = 'task-summary'),
                html.Div(id = 'query-summary')
                        ], style = {'margin-left': '10%','margin-right': '10%'}),
    html.Hr(),
    html.Div([
            html.H3('Search Results'),
            html.Div(id = 'task-results'),
            html.Div(id = 'query-results')
            ], id = 'search-results-main', style = {'margin-left': '10%','margin-right': '10%'}),
    html.Hr(),
])

@app.callback(
    Output('sub-task-dropdown', 'options'),
    [Input('task-dropdown', 'value')])
def set_subtask_options(selected_task):
    if selected_task != None:
        dff = df[df['Task Name'] == selected_task]
        options = dff['Sub-tasks'].unique().tolist()
        return [{'label': i, 'value': i} for i in options]
    else:
        return [{'label': i, 'value': i} for i in []]
    
@app.callback(
    Output('sub-task-dropdown', 'value'),
    [Input('sub-task-dropdown', 'options')])
def set_subtask_value(available_options):
    if available_options != []:
        return available_options[0]['value']
    else:
        return ''
    
@app.callback(
    Output('task-summary', 'children'),
    [Input('sub-task-dropdown', 'value')])
def update_taks_summary(value):
    if value != '':
        dff = df[df['Sub-tasks'] == value]
        return _output(dff['Output'].tolist())[0]
    else:
        return ''


@app.callback(
    Output('task-results', 'children'),
    [Input('sub-task-dropdown', 'value')])
def update_taks_results(value):
    if value != '':
        dff = df[df['Sub-tasks'] == value]
        return generate_table(dff)
    else:
        return ''
    
@app.callback(
        Output('query-results', 'children'),
         [Input('submit-button-state', 'n_clicks')],
         [State('general-search', 'value')]
         )
def populate_search_results(n_clicks, value):
    if value != '':
        query = value
        response = requests.post("https://nlp.biano-ai.com/develop/test", json={"texts": [query]})
        predictions = response.json()['predictions']
        pred_df = pd.DataFrame(predictions[0])
        pred_df.columns = ['Distance', 'Document ID', 'Output', 'Title', 'URL']
        return generate_table(pred_df)
    else:
        return ''

@app.callback(
        Output('query-summary', 'children'),
         [Input('submit-button-state', 'n_clicks')],
         [State('general-search', 'value')]
         )
def generate_search_summary(n_clicks, value):
    if value != '':
        query = value
        response = requests.post("https://nlp.biano-ai.com/develop/test", json={"texts": [query]})
        predictions = response.json()['predictions']
        pred_df = pd.DataFrame(predictions[0])
        return _output(pred_df['text'].tolist())[0]
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)
