from IPython.core.display import display, HTML
%%capture



!wget "https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz"

!tar -xvf openjdk-11.0.2_linux-x64_bin.tar.gz



!export JAVA_HOME='/kaggle/working/jdk-11.0.2/'

!export PATH='/kaggle/working/jdk-11.0.2/bin':$PATH
# %%capture

!mkdir -p /kaggle/working/jdk-11.0.2/jre/lib/amd64/server/

!ln -s /kaggle/working/jdk-11.0.2/lib/server/libjvm.so /kaggle/working/jdk-11.0.2/jre/lib/amd64/server/libjvm.so
# !ls -la /kaggle/working/jdk-11.0.2/lib/server/

# !ls -la /kaggle/working/jdk-11.0.2/jre/lib/amd64/server/
import json

import os



# linux

os.environ["JAVA_HOME"] = "/kaggle/working/jdk-11.0.2/"
%%capture

!pip install pyserini==0.8.1.0

!pip install transformers
%%capture

!wget https://www.dropbox.com/s/j1epbu4ufunbbzv/lucene-index-covid-2020-03-27.tar.gz

!tar xvfz lucene-index-covid-2020-03-27.tar.gz
!du -h lucene-index-covid-2020-03-27
import torch

import numpy

from tqdm import tqdm

from transformers import *

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)

# model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')
tokenizer = AutoTokenizer.from_pretrained('../input/scibertcord19checkpoint950000/checkpoint-950000', do_lower_case=False)

model = AutoModel.from_pretrained('../input/scibertcord19checkpoint950000/checkpoint-950000')
COVID_INDEX = 'lucene-index-covid-2020-03-27/'



def show_query(query):

    """HTML print format for the searched query"""

    return HTML('<br/><div style="font-family: Times New Roman; font-size: 20px;'

                'padding-bottom:12px"><b>Query</b>: '+query+'</div>')



def show_document(idx, doc):

    """HTML print format for document fields"""

    have_body_text = 'body_text' in json.loads(doc.raw)

    body_text = ' Full text available.' if have_body_text else ''

    return HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px">' + 

               f'<b>Document {idx}:</b> {doc.docid} ({doc.score:1.2f}) -- ' +

               f'{doc.lucene_document.get("authors")} et al. ' +

             # f'{doc.lucene_document.get("journal")}. ' +

             # f'{doc.lucene_document.get("publish_time")}. ' +

               f'{doc.lucene_document.get("title")}. ' +

               f'<a href="https://doi.org/{doc.lucene_document.get("doi")}">{doc.lucene_document.get("doi")}</a>.'

               + f'{body_text}</div>')



def show_query_results(query, searcher, top_k=10):

    """HTML print format for the searched query"""

    hits = searcher.search(query)

    display(show_query(query))

    for i, hit in enumerate(hits[:top_k]):

        display(show_document(i+1, hit))

    return hits[:top_k]   
from pyserini.search import pysearch



searcher = pysearch.SimpleSearcher(COVID_INDEX)

query = ('these differences reside in the molecular structure of spike proteins and some other factors. Which receptor combination(s) will cause maximum harm')

hits = show_query_results(query, searcher, top_k=10)
def extract_scibert(text, tokenizer, model):

    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]



    n_chunks = int(numpy.ceil(float(text_ids.size(1))/510))

    states = []

    

    for ci in range(n_chunks):

        text_ids_ = text_ids[0, 1+ci*510:1+(ci+1)*510]            

        text_ids_ = torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])

        if text_ids[0, -1] != text_ids[0, -1]:

            text_ids_ = torch.cat([text_ids_, text_ids[0,-1].unsqueeze(0)])

        

        with torch.no_grad():

            state = model(text_ids_.unsqueeze(0))[0]

            state = state[:, 1:-1, :]

        states.append(state)



    state = torch.cat(states, axis=1)

    return text_ids, text_words, state[0]
query_ids, query_words, query_state = extract_scibert(query, tokenizer, model)
ii = 0

doc_json = json.loads(hits[ii].raw)



paragraph_states = []

for par in tqdm(doc_json['body_text']):

    state = extract_scibert(par['text'], tokenizer, model)

    paragraph_states.append(state)
def cross_match(state1, state2):

    state1 = state1 / torch.sqrt((state1 ** 2).sum(1, keepdims=True))

    state2 = state2 / torch.sqrt((state2 ** 2).sum(1, keepdims=True))

    sim = (state1.unsqueeze(1) * state2.unsqueeze(0)).sum(-1)

    return sim
sim_matrices = []

for pid, par in tqdm(enumerate(doc_json['body_text'])):

    sim_score = cross_match(query_state, paragraph_states[pid][-1])

    sim_matrices.append(sim_score)
paragraph_relevance = [torch.max(sim).item() for sim in sim_matrices]



# Select the index of top 5 paragraphs with highest relevance

rel_index = numpy.argsort(paragraph_relevance)[-5:][::-1]
def show_sections(section, text):

    """HTML print format for document subsections"""

    return HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px; margin-left: 15px">' + 

        f'<b>{section}</b> -- {text.replace(" ##","")} </div>')



display(show_query(query))

display(show_document(ii, hits[ii]))

for ri in numpy.sort(rel_index):

    display(show_sections(doc_json["body_text"][ri]['section'], " ".join(paragraph_states[ri][1])))
def highlight_paragraph(ptext, rel_words, max_win=10):

    para = ""

    prev_idx = 0

    for jj in rel_words:

        

        if prev_idx > jj:

            continue

        

        found_start = False

        for kk in range(jj, prev_idx-1, -1):

            if ptext[kk] == "." and (ptext[kk+1][0].isupper() or ptext[kk+1][0] == '['):

                sent_start = kk

                found_start = True

                break

        if not found_start:

            sent_start = prev_idx-1

            

        found_end = False

        for kk in range(jj, len(ptext)-1):

            if ptext[kk] == "." and (ptext[kk+1][0].isupper() or ptext[kk+1][0] == '['):

                sent_end = kk

                found_end = True

                break

                

        if not found_end:

            if kk >= len(ptext) - 2:

                sent_end = len(ptext)

            else:

                sent_end = jj

        

        para = para + " "

        para = para + " ".join(ptext[prev_idx:sent_start+1])

        para = para + " <font color='blue'>"

        para = para + " ".join(ptext[sent_start+1:sent_end])

        para = para + "</font> "

        prev_idx = sent_end

        

    if prev_idx < len(ptext):

        para = para + " ".join(ptext[prev_idx:])



    return para
display(show_query(query))



display(show_document(ii, hits[ii]))



for ri in numpy.sort(rel_index):

    sim = sim_matrices[ri].data.numpy()

    

    # Select the two highest scoring words in the paragraph

    rel_words = numpy.sort(numpy.argsort(sim.max(0))[-2:][::-1])

    p_tokens = paragraph_states[ri][1]

    para = highlight_paragraph(p_tokens, rel_words)

    display(show_sections(doc_json["body_text"][ri]['section'], para))