import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Import HTML library for displaying results nicely

from IPython.core.display import display, HTML

# json reader

import json

# tqdm progress meter

from tqdm import tqdm



# huggingface transformers

!pip install transformers

import torch

from transformers import DistilBertTokenizer, DistilBertModel

from transformers import BartTokenizer, BartForConditionalGeneration
!curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz

!mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz

!update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1

!update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java

os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2"
!pip install pyserini==0.8.1.0

from pyserini.search import pysearch

# import lucene index (thanks, anserini team!)

!wget https://www.dropbox.com/s/d6v9fensyi7q3gb/lucene-index-covid-2020-04-03.tar.gz

!tar xvfz lucene-index-covid-2020-04-03.tar.gz
COVID_INDEX = 'lucene-index-covid-2020-04-03/'



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



searcher = pysearch.SimpleSearcher(COVID_INDEX)
# Enter query here

query = ('COVID-19 incubation period in humans')



hits = show_query_results(query, searcher, top_k=10)
model_version = 'distilbert-base-uncased'

do_lower_case = True

model = DistilBertModel.from_pretrained(model_version)

tokenizer = DistilBertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
def extract_distilbert(text, tokenizer, model):

    # Convert text to IDs with special tokens specific to the distilBERT model

    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]



    n_chunks = int(np.ceil(float(text_ids.size(1))/510))

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
query_ids, query_words, query_state = extract_distilbert(query, tokenizer, model)
my_fav_hit = 6

doc_json = json.loads(hits[my_fav_hit].raw)



paragraph_states = []

for par in tqdm(doc_json['body_text']):

    state = extract_distilbert(par['text'], tokenizer, model)

    paragraph_states.append(state)
# Compute similarity given the extracted states from sciBERT

def cross_match(state1, state2):

    state1 = state1 / torch.sqrt((state1 ** 2).sum(1, keepdims=True))

    state2 = state2 / torch.sqrt((state2 ** 2).sum(1, keepdims=True))

    sim = (state1.unsqueeze(1) * state2.unsqueeze(0)).sum(-1)

    return sim
# Compute similarity for each paragraph

sim_matrices = []

for pid, par in tqdm(enumerate(doc_json['body_text'])):

    sim_score = cross_match(query_state, paragraph_states[pid][-1])

    sim_matrices.append(sim_score)
paragraph_relevance = [torch.max(sim).item() for sim in sim_matrices]



# Select the index of top 5 paragraphs with highest relevance

rel_index = np.argsort(paragraph_relevance)[-5:][::-1]
def show_sections(section, text):

    """HTML print format for document subsections"""

    return HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px; margin-left: 15px">' + 

        f'<b>{section}</b> -- {text.replace(" ##","")} </div>')



display(show_query(query))

display(show_document(my_fav_hit, hits[my_fav_hit]))

for ri in np.sort(rel_index):

    display(show_sections(doc_json["body_text"][ri]['section'], " ".join(paragraph_states[ri][1])))
text = []

for ri in np.sort(rel_index):

    text = text + paragraph_states[ri][1]
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'



sum_tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')

sum_model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')

sum_model.to(torch_device)

# set to evaluation mode for speed and memory saving

sum_model.eval()



article_input_ids = sum_tokenizer.batch_encode_plus([tokenizer.convert_tokens_to_string(text)], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)



summary_ids = sum_model.generate(article_input_ids,

                             num_beams=4,

                             length_penalty=2.0,

                             max_length=1000,

                             no_repeat_ngram_size=3)



summary_txt = sum_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)



HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px; margin-left: 15px">' + 

        f'<p><b>Query:</b> {query}</p> &nbsp; <p><b>Summary of results:</b> {summary_txt}</div></p>')