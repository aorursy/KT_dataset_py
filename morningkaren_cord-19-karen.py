import numpy as np 
import pandas as pd 
import os
import json
from tqdm.notebook import tqdm
from pprint import pprint
from copy import deepcopy
df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
biorxiv_dir = '../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))
comm_use = '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json'
filenames = os.listdir(comm_use)
print("Number of articles retrieved from comm_use:", len(filenames))
custom_license = '../input/CORD-19-research-challenge/custom_license/custom_license/pdf_json'
filenames=os.listdir(custom_license)
print("Number of articles retrieved from custom_license:", len(filenames))
all_files = []

for filename in filenames:
    filename = custom_license + '/'+filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)
file = all_files[0]
print("Dictionary keys:", file.keys())
pprint(file['abstract'])
print("body_text type:", type(file['body_text']))
print("body_text length:", len(file['body_text']))
print("body_text keys:", file['body_text'][0].keys())
print("body_text content:")
pprint(file['body_text'][:2], depth=3)
texts = [(di['section'], di['text']) for di in file['body_text']]
texts_di = {di['section']: "" for di in file['body_text']}
for section, text in texts:
    texts_di[section] += text

pprint(list(texts_di.keys()))
body = ""

for section, text in texts_di.items():
    body += section
    body += "\n\n"
    body += text
    body += "\n\n"

print(body[:3000])
print(all_files[0]['metadata'].keys())
print(all_files[0]['metadata']['title'])
authors = all_files[0]['metadata']['authors']
pprint(authors[:3])
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


for author in authors:
    print("Name:", format_name(author))
    print("Affiliation:", format_affiliation(author['affiliation']))
    print()
authors = all_files[4]['metadata']['authors']
print("Formatting without affiliation:")
print(format_authors(authors, with_affiliation=False))
print("\nFormatting with affiliation:")
print(format_authors(authors, with_affiliation=True))
bibs = list(file['bib_entries'].values())
pprint(bibs[:2], depth=4)
format_authors(bibs[1]['authors'], with_affiliation=False)
bib_formatted = format_bib(bibs[:5])
print(bib_formatted)
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
clean_df.to_csv('custome_license_clean.csv', index=False)
biorxiv = pd.read_csv('biorxiv_clean.csv')[['paper_id', 'title', 'authors', 'abstract']]
biorxiv.to_csv('biorxiv_abstract.csv', index=False)



comm_use = '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset'
filenames = os.listdir(comm_use)
print("Number of articles retrieved from comm_use:", len(filenames))
custom_license = '../input/CORD-19-research-challenge/custom_license/custom_license'
filenames=os.listdir(custom_license)
print("Number of articles retrieved from custom_license:", len(filenames))
noncomm_use = '../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset'
filenames=os.listdir(noncomm_use)
print("Number of articles retrieved from noncomm_use:", len(filenames))
!pip install -U bert-serving-server bert-serving-client
!pip install bert-serving-server
!pip install bert-serving-client
import subprocess
bert_command = 'bert-serving-start -model_dir ../input/bert-model/uncased_L-12_H-768_A-12'
process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)
from bert_serving.client import BertClient
bc = BertClient()
embeddings = bc.encode(['Embed a single sentence', 
                        'Can it handle periods? and then more text?', 
                        'how about periods.  and <p> html stuffs? <p>'])
embeddings.shape
