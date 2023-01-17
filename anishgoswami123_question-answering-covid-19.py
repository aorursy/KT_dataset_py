# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import json
import os
#function for taking in a directory, returning a list of all dictionaries from directory
def list_of_dictionaries(directory):
    
    #getting list of filenames
    filenames = [pos_json for pos_json in os.listdir(directory) if pos_json.endswith('.json')]
    print('number of dictionaries:',len(filenames))
    
    #opening each .json file and putting each dictionary and putting it into a list
    dictionary_list = []
    for filename in filenames:
        json_file = open(f'{directory}{filename}')
        dictionary_list.append(json.load(json_file))
        json_file.close()
    print(len(dictionary_list))
    return dictionary_list
#function for saving list of dictionaries into a json file
def save_data(filename, data):
    with open(f'{filename}.json', 'w') as json_file:
        json.dump(data, json_file)
#combining all the lists of dictionaries into one giant list
full_data = []

biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'
for dictionary in list_of_dictionaries(directory=biorxiv_dir):
    full_data.append(dictionary)

comm_use_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/'
for dictionary in list_of_dictionaries(directory=comm_use_dir):
    full_data.append(dictionary)

noncomm_use_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/'
for dictionary in list_of_dictionaries(directory=noncomm_use_dir):
    full_data.append(dictionary)
    
custom_license_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/'
for dictionary in list_of_dictionaries(directory=custom_license_dir):
    full_data.append(dictionary)
#saving list of dictionaries to one file so I don't have to do this over and over
save_data(filename='real_full_data', filetype='.json', data=full_data)
#function to return data from .json
def read_file(filename, num_entries):
    data=[]
   
    # will make function support more file types later
    with open(f'{filename}{filetype}', 'r') as json_file:
        data = json.load(json_file)
    
    if num_entries == 'all':
        return data
    
    return data[:num_entries]
#functions for getting import information from a dictionary
def get_title(data):
    return data['metadata']['title']

def get_abstract(data):
    abstract = []
    #check to make sure abstract is not empty
    if data['abstract']:
        abstract = data['abstract'][0]['text']
        return abstract
    
    else:
        return abstract

def get_author(data):
    authors = data['metadata']['authors']
    author_list= []
    
    for author in authors:
        #checking to make sure there is not garbage in author section
        if author['first'] and author['last']:
            author_list.append(" ".join([author['first'], author['last']]))
    
    return author_list

def get_body_text_paragraphs(data):
    #each paragraph has a heading which is denoted as section
    body_text = ''
    section = ''
    full_text = data['body_text']
    
    for paragraph in full_text:
        yield paragraph['text']#, paragraph['section']

from nltk import sent_tokenize

def get_section(data):
    #each paragraph has a heading which is denoted as section
    body_text = ''
    section = ''
    full_text = data['body_text']
    
    for paragraph in full_text:
        yield paragraph['section']#, paragraph['section']

def get_full_body_text(data):
    full_text = ''
    for paragraph in data['body_text']:
        full_text += paragraph['text']
    
    return full_text
#takes list of dictionaries and returns dataframes
def create_dataframe(data):
    full_table = []
    for index, paper in enumerate(data):
        temp_list = []
        temp_list.append(get_title(paper))
        temp_list.append(get_author(paper))
        temp_list.append(get_abstract(paper))
        temp_list.append(get_full_body_text(paper))
        body_text = get_body_text_paragraphs(paper)
        section = get_section(paper)
        temp_list.append(list(body_text))
        temp_list.append(list(section))
        
        full_table.append(temp_list)
    
    #creating pandas dataframe because it will be easy to work with
    dataframe = pd.DataFrame(full_table, columns=['title', 'author', 'abstract','full_text', 'text_paragraphs','section'])
    return dataframe
#loading in files
unparsed_data = read_file(filename='/kaggle/input/loading-json-s/real_full_data', filetype='.json', num_entries='all')

#creating dataframe and saying 
dataframe = create_dataframe(unparsed_data)
dataframe.to_csv('dataframe.csv', index=False, header=True)
data_frame_filepath = '/kaggle/input/parsing-jsons/dataframe.csv'
#some parts need to be kept as lists beacuse reading from csv will by default turn lists into strings
keep_as_lists = {
    'text_paragraphs': eval,
    'sentences': eval,
    'section': eval
}
corona_data = pd.read_csv(data_frame_filepath, converters=keep_as_lists)
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Doc2Vec

#function returning tagged docouments
def get_tagged_documents(dataframe):
    # probably should process all data at the same time because vectorized process is much better optimizied then iterating through
    # loop but I couldn't figure out how to do that
    # converted dataframe columns to lists because lists are much faster to iterate over
    unprocessed_texts = dataframe['full_text'].tolist()
    tags = dataframe['title'].tolist()
    i=0
    for paper, tag in zip(unprocessed_texts, tags):
        #cleaning up text and splitting it into tokens
        tokens = gensim.utils.simple_preprocess(paper)
        i+=1

        yield gensim.models.doc2vec.TaggedDocument(tokens, [f"PAPER_{i}"]) 
train_corpus1 = list(create_paper_vectors(corona_data)) 
paper_model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=50, epochs=10, workers=4) 
#model needs a vocabulary of all words in corpus before it can be trained
paper_model.build_vocab(train_corpus1)

for epoch in range(paper_model.epochs): 
    paper_model.train(train_corpus1, total_examples=paper_model.corpus_count, epochs=1) 
    print('epoch_completed')

paper_model.save('paperDoc2Vec.bin')
from gensim.models.doc2vec import Doc2Vec

#loading in dataframe so that i can find the titles and abstracts from tags
corona_file = '/kaggle/input/covid-dataframe/dataframe.csv'
corona_data = pd.read_csv(corona_file, usecols=['title','abstract'])
#loading in document embeddings 
paper_model = Doc2Vec.load('/kaggle/input/paper-vectors-1/paperDoc2Vec.bin')
#functions to convert tags to paper titles and abstracts
def get_title(dataframe, index):
    return dataframe['title'][index-1]
def get_abstract(dataframe, index):
    return dataframe['abstract'][index-1]
#most_similar function only returns tags which arent very helpful, this function returns the titles and abstracts in a dataframe
def similar_vectors(words, n_results, model, dataframe):
    query = paper_model.infer_vector(words)
    #function mentioned earlier
    ids = model.docvecs.most_similar([query], topn=n_results)
    similarity_score = []
    titles = []
    full_data = []
    for id in ids:
        temp_list = []
        index = int(id[0].replace("PAPER_",""))
        title = get_title(dataframe, index)
        abstract = get_abstract(dataframe, index)
        temp_list.append(title)
        temp_list.append(abstract)
        temp_list.append(id[1])
        full_data.append(temp_list)

    return pd.DataFrame(data=full_data, columns=['title', 'abstract', 'simililarity','paragraphs'])