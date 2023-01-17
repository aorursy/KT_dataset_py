# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from pprint import pprint
from copy import deepcopy
import collections
from collections import Counter 
from typing import List, Dict, Any, Generator
from collections import OrderedDict
import gensim.downloader as api
from IPython.display import Markdown, display
from rake_nltk import Rake
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
import en_core_web_md
word_vectors = api.load("glove-wiki-gigaword-100")
nlp = en_core_web_md.load()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


#for dirname, _, filenames in os.walk('/kaggle/input/'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def get_keyword(task:str):
    # Use Rake from nltk to extract keyword from task question
    stops_words=stopwords.words('english')+['virus', 'know', 'known']
    r = Rake(stopwords=stops_words, min_length=2) # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(task)
    return r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.
    
    
def get_keyword2(task:str):
    # Use spacy model to to extract keyword from task question
    result = []
    pos_tag = ['PROPN', 'NOUN'] 
    nlp.vocab["virus"].is_stop = True
    nlp.vocab["disease"].is_stop = True
    doc = nlp(task.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation or token.is_stop):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
                
    return result     
task='what we know about Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.'
print('key sentences extraction using rake_nltk: ')
print(get_keyword(task))

print('\nkeywords extraction using spacy: ')
print(get_keyword2(task))
def get_file_paths(data_path: str) -> List[str]:
    """ Return all JSON file from a folder """
    file_paths = [os.path.join(
        data_path, file_name) for file_name in os.listdir(
            data_path) if file_name.endswith(".json")]
    print(f"Found {len(file_paths)} files.")
    return file_paths

def read_file(file_path: str) -> Dict[str, Any]:
    """ Open JSON file and return dict() data """
    print(f"Reading file {file_path}.")
    with open(file_path, "r") as handler:
        json_data = json.loads(handler.read(), object_pairs_hook=OrderedDict)
    return json_data
def get_paths(json_data: Dict[str, Any]) -> List[List[str]]:
    """ Return all paths defined by JSON keys """
    paths = []
    if isinstance(json_data, collections.MutableMapping):
        for k, v in json_data.items():
            paths.append([k])
            paths += [[k] + x for x in get_paths(v)]
    elif isinstance(json_data, collections.Sequence) and not isinstance(json_data, str):
        for i, v in enumerate(json_data):
            paths.append([i])
            paths += [[i] + x for x in get_paths(v)]
    return paths

def extract_section_text_from_paths(paths: List[List[str]], section: str) -> Generator[List[str], None, None]:
    """ Yield paths of abstract, body_text, ref_entries or back_matter """
    section_paths = (path for path in paths if path[0] == section and path[-1] == "text")
    return section_paths

def extract_text_from_path(paths: List[str], data: Dict[str, Any]) -> Generator[str, None, None]:
    """ Use paths (list of keys to follow) to yield texts from JSON """
    for path in paths:
        node_data = data
        for key in path:
            node_data = node_data[key]
        if len(node_data) > 20:
            yield(node_data)
def preprocess_paragraphs(dict_of_paragraphs: Dict[str, Generator[str, None, None]]):

    print("TEXT PRE-PROCESSINGTO BE IMPLEMENTED")
    abstract=[]
    body=[]
    legends=[]
    #print(dict_of_paragraphs)
    for k, l in dict_of_paragraphs.items():  
        #print(f"== {k} ==")  # Abstract, body or figures legend as keys
        for s in l:  # Generatos of paragraphs as values
            if k=='abstract':
                abstract.append(s) 
            elif k=='body':
                body.append(s)
            else:
                legends.append(s)
                
    #print(len(abstract))  # Sentences as str
    #print(len(body))   
    #print(len(legends))
            
    return [abstract, body, legends]
data_path=['/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json',
'/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json',
'/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json',
'/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json',
'/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json']
files_paths=[]
for fold in data_path:
    files_paths= files_paths+ get_file_paths(fold) 

print(len(files_paths))    
    
list_file=[]
list_titles=[]
list_abstract=[]
list_body=[]
list_legends=[]
for file_path in files_paths:

    # Read JSON and extract key paths from JSON root to leaves
    json_file_content: Dict[str, Any] = read_file(file_path)
    list_titles.append(json_file_content['metadata']['title'])     
    paths: List[List[str]] = get_paths(json_data=json_file_content)

    # Extract paths in JSON data leading to a text from different sections
    abstracts_paths = extract_section_text_from_paths(paths=paths, section="abstract")
    body_paths = extract_section_text_from_paths(paths=paths, section="body_text")
    figures_legends_paths = extract_section_text_from_paths(paths=paths, section="ref_entries")

    # Get paragraphs (defined by the JSON structure) from these parts
    paragraphs = {
        "abstract": extract_text_from_path(paths=abstracts_paths, data=json_file_content),
        "body": extract_text_from_path(paths=body_paths, data=json_file_content),
        "legends": extract_text_from_path(paths=figures_legends_paths, data=json_file_content)}

    # Preprocess them
    preprocessed_paragraphs: Dict[str, List[str]] = preprocess_paragraphs(dict_of_paragraphs=paragraphs)
    list_file.append(file_path.split('/')[-1].split('.')[0])
    list_abstract.append(' '.join([str(elem) for elem in preprocessed_paragraphs[0]]))
    list_body.append(' '.join([str(elem) for elem in preprocessed_paragraphs[1]]))
    list_legends.append(' '.join([str(elem) for elem in preprocessed_paragraphs[2]]))    
paper_table = pd.DataFrame(list(zip(list_file, list_titles, list_abstract, list_body, list_legends)), 
               columns =['ID', 'title','abstract','body', 'legends'])
del(list_file)
del(list_titles)
del(list_abstract)
del(list_body)
del(list_legends)
paper_table.head()
def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict

def filter_similar_paper(df, task):
    simil_dict={}
    for id in df['ID']:
        #print(id)
        sentence=df[df['ID']==id]['abstract'].values[0]
        simil_dict[id]= word_vectors.wmdistance(sentence, task)
    simil_dict=filterTheDict(simil_dict, lambda elem: elem[1] <2)
        
    return df[df['ID'].isin(list(simil_dict.keys())) ]
def get_task_information(task:str, paper_table):
    # function that display for each extract word  the part of the paper_id and 
    #the part of the abstract that contains that word
    
    paper_table= filter_similar_paper(paper_table, task)
    print(paper_table.shape)
    
    #key_terms=get_keyword(task)+get_keyword2(task)
    key_terms=get_keyword(task)
    
    index=0
    for abstract in paper_table.iloc[:,3]:
        term_in=False
        for terms in key_terms:
            #print(terms)
            terms2=terms
            if terms2 in abstract.lower() and ("covid-19" in abstract.lower() or "coronavirus" in abstract.lower()): 
                term_in=True
        if term_in:
            if len(paper_table.iloc[index,1])>0:
                display(Markdown('<b>'+"In the paper:"+' ' + paper_table.iloc[index,1]+'<b>'))
            else:
                display(Markdown('<b>'+"In the paper:"+' ' + paper_table.iloc[index,0]+'<b>'))
            
            #display(Markdown(df.iloc[index,2]))        
            #display(Markdown('<b>'+'In conclusion:'+'<b>'))
            sentence3="<ul>"
            for sentence in abstract.split('. '):
                sentence2=sentence.lower()
                matched2=False
                for terms in key_terms: 
                    terms2=terms
                    if terms2 in sentence2: 
                        matched2=True
                        sentence2=sentence2.replace(terms, '<b>'+terms+'</b>')
                        #sentence2='<b>'+sentence2+'<b>'
                if matched2: 
                    sentence3=sentence3+"<li>"+sentence2.replace("\n","")+"</li>"
            display(Markdown(sentence3+"</ul>"))       
        
        index+=1
get_task_information("what we know about  incubation periods", paper_table)