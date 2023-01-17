# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input/'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import json
import warnings
warnings.simplefilter('ignore')

JSON_PATH = '/kaggle/input/CORD-19-research-challenge/document_parses/pdf_json'


#json_files = [pos_json for pos_json in os.listdir(JSON_PATH) if pos_json.endswith('.json')]
# take all json files available
json_files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.json'):
            json_files.append(os.path.join(dirname, filename))
len(json_files)
json_files[118720]
corpus_abstract = {}
corpus_body = {}
# corpus_id = []
count=-1
# loop through the files
for jfile in json_files[::]:
    count=count+1
    abstract=[]
    body=[]
    # for each file open it and read as json
    with open(os.path.join(JSON_PATH, jfile)) as json_file:
#         print(count)
        covid_json = json.load(json_file)
        try:
            if (covid_json['abstract']==[] or covid_json['body_text']==[] or covid_json['paper_id']==[]):
                count=count-1
                continue
#             print("rama")
            key=covid_json['paper_id']
#             print("krishna")
            for item in covid_json['abstract']:
                abstract.append(item['text'])
            for item in covid_json['body_text']:
                body.append(item['text'])
            corpus_abstract[key]=abstract
            corpus_body[key]=body
        except:
            break
print("Corpus size = %d"%(len(corpus_body)))
print("Corpus size = %d"%(len(corpus_abstract)))
list(corpus_abstract.items())[:2]
list(corpus_body.items())[:2]
from transformers import pipeline
nlp = pipeline('question-answering')
def askmeAnything(que,con):
    return nlp({'question': que, 'context':con})['answer']
sent='Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic'
key_words=['copper','stainless','steel','plastic']
key_words=['silver']
key_words=['asymptomatic', 'shedding' ,'transmission','children']
import nltk
zero=list(corpus_abstract.values())[0:1]
print(zero[0][0])
cnt=0
relevant_ids=[]
for t in corpus_abstract.keys():
    if cnt==20:
        break
        
    for key in key_words:
        flag=0
#         print(key)
        ab=corpus_abstract[t]
        for a in ab:
            if key in a:
                relevant_ids.append(t)
                flag=1
                break
        if flag==1:
            print(cnt)
            cnt=cnt+1
print(relevant_ids)
con=[]
for id in relevant_ids:
    con.append('.'.join(corpus_abstract[id]))
con='.'.join(con)
print(con)
que='How is the prevalence of asymptomatic shedding and transmission particularly children?'
askmeAnything(que,con)

import tensorflow_hub as hub
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
embed([sent])
import numpy as mp
import pandas as pd
titles=list(corpus_abstract.values())
embed_vectors=embed(titles)
titles=corpus_abstract
embed_vectors=embed(titles[:100].values)['outputs'].numpy()

sentence=sent
print("Find similar research papers for :")
print(sentence)

similarity_matrix=prepare_similarity(embed_vectors)
similar=get_top_similar(sentence,sentence_list,similarity_matrix,6)
def prepare_similarity(vectors):
    similarity=cosine_similarity(vectors)
    return similarity

def get_top_similar(sentence, sentence_list, similarity_matrix, topN):
    # find the index of sentence in list
    index = sentence_list.index(sentence)
    # get the corresponding row in similarity matrix
    similarity_row = np.array(similarity_matrix[index, :])
    # get the indices of top similar
    indices = similarity_row.argsort()[-topN:][::-1]
    return [(i,sentence_list[i]) for i in indices]

titles=all_sources['title'].fillna("Unknown")
embed_vectors=embed(titles[:100].values)['outputs'].numpy()
sentence_list=titles.values.tolist()
sentence=titles.iloc[5]
print("Find similar research papers for :")
print(sentence)

similarity_matrix=prepare_similarity(embed_vectors)
similar=get_top_similar(sentence,sentence_list,similarity_matrix,6)