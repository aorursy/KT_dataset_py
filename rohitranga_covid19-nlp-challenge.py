import requests
import json

headers = {'accept': 'application/json','Content-Type': 'text/plain'}
params = (('annotationTypes', '*'),('language', 'en'))

def get_json_object(text):
    return requests.post('<our url>', headers=headers, params=params, data=text).json()

# Output: json string
def get_json_str(json_obj):
    return json.dumps(json_obj)

# Output: beautified json string
def get_pretty_json(json_str):
    return json.dumps(json_str, indent=4)

# Output: List of themes
def get_themes(text):
    json_obj = get_json_object(text)
    json_array = json_obj["annotationDtos"]
    return json_array[-1]["themes"]
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def lst_to_str(word_list):
    return ' '.join(word_list).strip()
import numpy as np
import json
import os
import csv
import time
from nltk.tokenize import sent_tokenize

root = '/kaggle/input/dataset/CORD-19-research-challenge/'
folders = ['biorxiv_medrxiv/biorxiv_medrxiv/', 'comm_use_subset/comm_use_subset/', 
           'noncomm_use_subset/noncomm_use_subset/', 'custom_license/custom_license/']


def collect_sentences():
    index_in_docs = 0
    num_files_processed = 0
    sentences_np_array = np.empty(100000000, dtype=object)

    start = time.time()
    for folder in folders:
        for filename in os.listdir(root+folder):
            if filename.endswith(".json"): 
                input_file_path = root+folder+filename
                with open(input_file_path) as f:
                    data = json.load(f)

                    # Collect abstract sentences
                    abstracts = data['abstract']
                    for content in abstracts:
                        abstract_para = content['text']
                        sentences = sent_tokenize(abstract_para)
                        for sentence in sentences:
                            sentences_np_array[index_in_docs] = sentence
                            index_in_docs += 1

                    # Collect body sentences
                    body_texts = data['body_text']
                    for content in body_texts:
                        body_para = content['text']
                        sentences = sent_tokenize(body_para)
                        for sentence in sentences:
                            sentences_np_array[index_in_docs] = sentence
                            index_in_docs += 1
                num_files_processed += 1            
                print('Num files processed: ' + str(num_files_processed))
                print('Time taken since beginning = ' + str(time.time()-start))
    np.save('sentences.npy', sentences_np_array) 
import json
import os
import csv
import time
from nltk.tokenize import sent_tokenize

root = '/kaggle/input/dataset/CORD-19-research-challenge/'
folders = ['biorxiv_medrxiv/biorxiv_medrxiv/', 'comm_use_subset/comm_use_subset/', 
           'noncomm_use_subset/noncomm_use_subset/', 'custom_license/custom_license/']

def collect_json_docs():
    docs = np.empty(100000000, dtype=np.object) 

    index_in_docs = 0
    num_files_processed = 0
    num_docs_collected = 0

    start = time.time()
    for folder in folders:
        for filename in os.listdir(root+folder):
            if filename.endswith(".json"): 
                input_file_path = root+folder+filename
                print(input_file_path)
                with open(input_file_path) as f:
                    data = json.load(f)

                    # Collect paper title
                    paper_title = data['metadata']['title']

                    # Collect authors' names
                    authors = data['metadata']['authors']
                    authors_names = []

                    for author in authors:
                        first_name = author['first']
                        middle_name = author['middle']
                        last_name = author['last']
                        author_name = first_name + ' ' + lst_to_str(middle_name) + ' ' + last_name
                        authors_names.append(author_name)

                    # Collect abstract sentences
                    abstracts = data['abstract']
                    for content in abstracts:
                        abstract_para = content['text']
                        section = content['section']
                        sentences = sent_tokenize(abstract_para)
    #                     para_themes = get_themes(abstract_para)
                        for sentence in sentences:
                            new_doc = {
                                "sentence": sentence,
                                "section": section,
                                "paper_title": paper_title,
                                "authors": authors_names,
                                "paragraph": abstract_para
    #                             "para_themes": para_themes
                            }
                            print(new_doc)
                            docs[index_in_docs] = new_doc
                            index_in_docs += 1
                            num_docs_collected += 1

                    # Collect body sentences
                    body_texts = data['body_text']
                    for content in body_texts:
                        body_para = content['text']
                        section = content['section']
                        sentences = sent_tokenize(body_para)
    #                     para_themes = get_themes(body_para)
                        for sentence in sentences:
                            new_doc = {
                                "sentence": sentence,
                                "section": section,
                                "paper_title": paper_title,
                                "authors": authors_names,
                                "paragraph": body_para
    #                             "para_themes": para_themes
                            }
                            print(new_doc)
                            docs[index_in_docs] = new_doc
                            index_in_docs += 1
                            num_docs_collected += 1
                num_files_processed += 1

    np.save('docs', docs) # by default, allow pickle = True
import numpy as np
docs = np.load('/kaggle/input/jsondocs/docs.npy', allow_pickle=True)
def collect_sentences():
    sentences=[]
    for jsonobject in docs:
        sentences.append(jsonobject['sentence'])
import tensorflow as tf
import time

 
def generate_embeddings():
    start = time.time()
    index=0
    batch_size = 3000
    num_rows=len(sentences)


    embeddings = np.empty((num_rows,512), dtype=np.float32)
    while index < num_rows:
        end_index = index+batch_size
        if end_index > num_rows:
            break
        embeddings[index:end_index,:] = embed(sentences[index:end_index])
        index += batch_size

    if index < num_rows:
        embeddings[index:num_rows,:] = embed(sentences[index:num_rows])

    np.save('embeddings/embeddings.npy', embeddings) # removed from working directory
!python -m pip install --upgrade faiss faiss-gpu
import numpy as np
import faiss

def create_index():
    embeddings = np.load('embeddings/embeddings.npy',mmap_mode='r')
    index = faiss.IndexScalarQuantizer(512,faiss.ScalarQuantizer.QT_6bit) 
    index.train(embeddings)
    index.add(embeddings)

    faiss.write_index(index, "vector_6.index")
import faiss
index = faiss.read_index("/kaggle/input/vector/vector_6.index", faiss.IO_FLAG_MMAP|faiss.IO_FLAG_READ_ONLY)  # load the index
import time
query = ["What has been published concerning research and development and evaluation efforts of vaccines and therapeutics for COVID-19?"]
query_vector = embed(query)
query_vector = np.asarray(query_vector, dtype=np.float32)

start = time.time()
D, I = index.search(query_vector, 10)  
!pip install json2html
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from json2html import *
from IPython.core.display import display, HTML
stopwords = set(STOPWORDS)

for id_index in I[0]:
    doc = docs[id_index]
    html = json2html.convert(doc)
    html = html.replace("<td>", "<td style='text-align:left'>")
    display(HTML(html))
    themes_list = doc['themes']
    final_theme_string = ''
    for theme in themes_list:
        words = theme.replace('-', ' ').split()
        t = '_'.join(words)
        final_theme_string = final_theme_string + ' ' + t
        
    # plot the WordCloud image  
    if doc['themes'] and doc['themes'][0]:
        wordcloud = WordCloud(width = 700,height = 200,stopwords = stopwords,min_font_size = 8, 
                              max_font_size=20, background_color='white', 
                              prefer_horizontal=1).generate(final_theme_string)
        plt.figure(figsize = (10, 10), linewidth=10, edgecolor="#04253a")
        plt.imshow(wordcloud, interpolation="bilinear") 
        plt.axis("off") 
        plt.show() 
    display(HTML("<hr style='height:3px; color:black'>"))
import pandas as pd
writer = pd.ExcelWriter('QueryResult1.xlsx', engine='xlsxwriter')

def writepaperdetails(query,retrivallines,subquestion):
    print(subquestion)
    print(retrivallines)
    Rpaper_title=[]
    Rsection=[]
    Rsentence=[]
    Rparagraph=[]
    Rthemes=[]
   
    for retrivalline in retrivallines:
        Rpaper_title.append(docs[retrivalline]['paper_title'])
        Rsection.append(docs[retrivalline]['section'])
        Rsentence.append(docs[retrivalline]['sentence'])
        Rparagraph.append(docs[retrivalline]['paragraph'])
        Rthemes.append(docs[retrivalline]['themes'])
       
    df = pd.DataFrame()
    dfquery=pd.DataFrame()
    df['PAPER_TITLE']=Rpaper_title
    df['SECTION']=Rsection
    df['SENTENCE']=Rsentence
    df['PARAGRAPH']=Rparagraph
    df['THEMES']=Rthemes
    dfquery['QUERY']=[query]
   
   
    dfquery.to_excel(writer, sheet_name="QUERY_"+str(subquestion))
    df.to_excel(writer, sheet_name="QUERY_"+str(subquestion),startrow=4)
def generate_excel_files(query, I):
    for i,q in enumerate(query):
        writepaperdetails(q,I.tolist()[i],i)
    writer.save()