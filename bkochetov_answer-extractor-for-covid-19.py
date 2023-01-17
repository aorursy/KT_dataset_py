!pip install cdqa
import numpy as np
import pandas as pd
import os
import json
import nltk
from math import log, sqrt
from collections import defaultdict
from copy import deepcopy
# Paths to json files
path_1 = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'
path_2 = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'
path_3 = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'
path_4 = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'

# List of folder names
folder_names = ['biorxiv_medrxiv','comm_use_subset']
folder_paths = [path_1, path_2]
# This piece of code was adopted from the original source at:
# https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv/notebook 

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
        else:# First, for each query the system arranges all the scientific papers within the corpus in the relevant order.
# Second, the system analize texts of top N the mosr relevant papers to answer to the query in the best way.
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

def load_files(dirname, filename=None):
    filenames = os.listdir(dirname)
    raw_files = []
    if filename:
        filename = dirname + filename
        raw_files = [json.load(open(filename, 'rb'))]
    else:
        #for filename in tqdm(filenames):
        for filename in filenames:
            filename = dirname + filename
            file = json.load(open(filename, 'rb'))
            raw_files.append(file)
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    #for file in tqdm(all_files):
    for file in all_files:
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
    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']
    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df = clean_df.drop(columns=['authors','affiliations','bibliography',
                                      'raw_authors','raw_bibliography'])
    return clean_df
def get_corpus(folder_paths = folder_paths):
    num_of_papers = {}
    corpus = pd.DataFrame(columns=['paper_id','title','abstract','text'])
    for i in range(len(folder_paths)):
        filenames = os.listdir(folder_paths[i])
        print('Reading {0} json files from folder {1} ...'.format(len(filenames), folder_names[i]))
        num_of_papers[folder_names[i]] = len(filenames)
        files = load_files(folder_paths[i])
        df = generate_clean_df(files)
        corpus = pd.concat([corpus, df], ignore_index=True, sort=False)
    print('Corpus includes {0} scientific articles.'.format(len(corpus)))
    return corpus, num_of_papers

corpus, num_of_papers = get_corpus()
# This processing algorithm can originaly be found at:
# https://github.com/nilayjain/text-search-engine

inverted_index = defaultdict(list)
num_of_documents = len(corpus)
vects_for_docs = []  # we will need nos of docs number of vectors, each vector is a dictionary
document_freq_vect = {}  # sort of equivalent to initializing the number of unique words to 0

# It updates the vects_for_docs variable with vectors of all the documents.
def iterate_over_all_docs():
    print('Processing corpus...')
    for i in range(num_of_documents):
        if np.mod(i, 1000) == 0:
            print('{0} of {1}'.format(str(i).zfill(len(str(num_of_documents))),num_of_documents))
        doc_text = corpus['title'][i] + ' ' + corpus['abstract'][i] + ' ' + corpus['text'][i]
        token_list = get_tokenized_and_normalized_list(doc_text)
        vect = create_vector(token_list)
        vects_for_docs.append(vect)
    print('{0} of {1}'.format(num_of_documents, num_of_documents))

def create_vector_from_query(l1):
    vect = {}
    for token in l1:
        if token in vect:
            vect[token] += 1.0
        else:
            vect[token] = 1.0
    return vect

def generate_inverted_index():
    count1 = 0
    for vector in vects_for_docs:
        for word1 in vector:
            inverted_index[word1].append(count1)
        count1 += 1

def create_tf_idf_vector():
    vect_length = 0.0
    for vect in vects_for_docs:
        for word1 in vect:
            word_freq = vect[word1]
            temp = calc_tf_idf(word1, word_freq)
            vect[word1] = temp
            vect_length += temp ** 2
        vect_length = sqrt(vect_length)
        for word1 in vect:
            vect[word1] /= vect_length

def get_tf_idf_from_query_vect(query_vector1):
    vect_length = 0.0
    for word1 in query_vector1:
        word_freq = query_vector1[word1]
        if word1 in document_freq_vect:
            query_vector1[word1] = calc_tf_idf(word1, word_freq)
        else:
            query_vector1[word1] = log(1 + word_freq) * log(
                num_of_documents)
        vect_length += query_vector1[word1] ** 2
    vect_length = sqrt(vect_length)
    if vect_length != 0:
        for word1 in query_vector1:
            query_vector1[word1] /= vect_length

def calc_tf_idf(word1, word_freq):
    return log(1 + word_freq) * log(num_of_documents / document_freq_vect[word1])

def get_dot_product(vector1, vector2):
    if len(vector1) > len(vector2):
        temp = vector1
        vector1 = vector2
        vector2 = temp
    keys1 = vector1.keys()
    keys2 = vector2.keys()
    sum = 0
    for i in keys1:
        if i in keys2:
            sum += vector1[i] * vector2[i]
    return sum

def get_tokenized_and_normalized_list(doc_text):
    tokens = nltk.word_tokenize(doc_text)
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in tokens:
        stemmed.append(ps.stem(words))
    return stemmed

def create_vector(l1):
    vect = {}  # this is a dictionary
    global document_freq_vect
    for token in l1:
        if token in vect:
            vect[token] += 1
        else:
            vect[token] = 1
            if token in document_freq_vect:
                document_freq_vect[token] += 1
            else:
                document_freq_vect[token] = 1
    return vect

def get_result_from_query_vect(query_vector1):
    parsed_list = []
    for i in range(num_of_documents - 0):
        dot_prod = get_dot_product(query_vector1, vects_for_docs[i])
        parsed_list.append((i, dot_prod))
        parsed_list = sorted(parsed_list, key=lambda x: x[1])
    return parsed_list

iterate_over_all_docs()
generate_inverted_index()
create_tf_idf_vector()
# The End-To-End Closed Domain Question Answering System is used here.
# It is available at: https://pypi.org/project/cdqa/

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline

download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
download_model(model='bert-squad_1.1', dir='./models')
def find_relevant_articles(query=None, top_n_papers=20, min_n_papers=3):
    if query == None:
        query = input('Please enter your query...')
    print('\n\n'+'*'*34+' PROCESSING NEW QUERY '+'*'*34+'\n')   
    query_list = get_tokenized_and_normalized_list(query)
    query_vector = create_vector_from_query(query_list)
    get_tf_idf_from_query_vect(query_vector)
    result_set = get_result_from_query_vect(query_vector)
    papers_info = {'query':query, 'query list':query_list, 'query vector':query_vector,
                   'id':[], 'title':[], 'abstract':[], 'text':[], 'weight':[], 'index':[]}
    for i in range(1, top_n_papers+1):
        tup = result_set[-i]
        papers_info['id'].append(corpus['paper_id'][tup[0]])
        papers_info['title'].append(corpus['title'][tup[0]])
        papers_info['abstract'].append(corpus['abstract'][tup[0]])
        papers_info['text'].append(corpus['text'][tup[0]])
        papers_info['weight'].append(tup[1])
        papers_info['index'].append(tup[0])
    colms = ['date', 'title', 'category', 'link', 'abstract', 'paragraphs']
    df = pd.DataFrame(columns=colms)
    for i in range(len(papers_info['text'])):
        papers_info['text'][i] = papers_info['text'][i].replace('\n\n', ' ')
        CurrentText = papers_info['text'][i]
        CurrentText = CurrentText.split('. ')
        #CurrentList = ["None", papers_info['title'][i], "None", "None", "None", CurrentText]
        CurrentList = ["None", papers_info['title'][i], "None", "None", papers_info['abstract'][i], CurrentText]
        CurrentList = np.array(CurrentList)
        CurrentList = CurrentList.reshape(1, CurrentList.shape[0])
        CurrentList = pd.DataFrame(data = CurrentList, columns=colms)
        df = pd.concat([df, CurrentList], ignore_index=True)
    df = filter_paragraphs(df)
    # Loading QAPipeline with CPU version of BERT Reader pretrained on SQuAD 1.1
    cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib')
    # Fitting the retriever to the list of documents in the dataframe
    cdqa_pipeline.fit_retriever(df=df)
    # Sending a question to the pipeline and getting prediction
    query = papers_info['query']
    prediction = cdqa_pipeline.predict(query=query)
    for i in range(top_n_papers):
        if papers_info['title'][i] == prediction[1]:
            pid = papers_info['id'][i]
    response = {query:{'id':pid,'title':prediction[1],'answer':prediction[0],'summary':prediction[2],
                       'important papers':{'id':papers_info['id'],'title':papers_info['title']}}}
    print('QUERY: {0}\n'.format(query))
    print('ANSWER MINED FROM PAPER: {0}\n'.format(prediction[0]))
    print('PAPER TITLE: {0}\n'.format(prediction[1]))
    print('PARAGRAPH IN PAPER: {0}\n'.format(prediction[2]))
    show_paper = np.min([min_n_papers, top_n_papers])
    print('\nTOP {0} MOST RELEVANT PAPERS RELATED TO THE QUERY:\n'.format(show_paper))
    for i in range(show_paper):
        print('PAPER #{0}. \nID: {1} \nTITLE: {2}\n'.format(i+1, papers_info['id'][i], papers_info['title'][i]))
    return response, papers_info, prediction, result_set, df
# List of queries
queries = ['What is range of incubation period for coronavirus SARS-CoV-2 COVID-19 in humans',
           'What is optimal quarantine period for coronavirus COVID-19',
           'What is effective quarantine period for coronavirus COVID-19',
           'What is percentage of death cases for coronavirus SARS-CoV-2 COVID-19',
           'What is death rate for coronavirus COVID-19 and air pollution',
           'At which temperature coronavirus COVID-19 can survive',
           'How long coronavirus SARS-CoV-2 can survive on plastic surface',
           'What are risk factors for coronavirus COVID-19',
           'What is origin of coronavirus COVID-19',
           'At which temperature coronavirus cannot survive']
for query in queries:
    response, papers_info, prediction, result_set, df = find_relevant_articles(query, top_n_papers=50)
query = 'What is coronavirus SARS-CoV-2'
find_relevant_articles(query=query, top_n_papers=50, min_n_papers=5);