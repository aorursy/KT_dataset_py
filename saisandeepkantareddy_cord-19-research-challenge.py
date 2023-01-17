import pandas as pd

import json

import re
# with open("../input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/252878458973ebf8c4a149447b2887f0e553e7b5.json", "r") as read_file:

#     data = json.load(read_file)
# data.keys()
# data['paper_id']
# data['metadata']['authors'][0].keys()
bio=pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')

pmc=pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv')

comm=pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv')

non_comm=pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv')
conn=pd.concat([bio,pmc,comm,non_comm])
conn.isna().sum()
#Finding any word contains transmission,incubation, environmental stability
conn_extracted=conn[conn.apply(lambda row: row.astype(str).str.contains('transmission|incubation|environmental stability').any(), axis=1)]
def clean_text(i):

    return re.sub(r"\n\n", " ", i)
conn_extracted['text']=conn_extracted['text'].apply(lambda x: clean_text(x))
conn_extracted.head(1)
meta=pd.read_csv('../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
meta.head()
meta['has_full_text'].value_counts()
# import os

# import json

# from pprint import pprint

# from copy import deepcopy



# import numpy as np

# import pandas as pd

# from tqdm.notebook import tqdm

# biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'

# filenames = os.listdir(biorxiv_dir)

# print("Number of articles retrieved from biorxiv:", len(filenames))
# all_files = []



# for filename in filenames:

#     filename = biorxiv_dir + filename

#     file = json.load(open(filename, 'rb'))

#     all_files.append(file)
# file = all_files[0]

# print("Dictionary keys:", file.keys())
# pprint(file['abstract'])
# print("body_text type:", type(file['body_text']))

# print("body_text length:", len(file['body_text']))

# print("body_text keys:", file['body_text'][0].keys())
# print("body_text content:")

# pprint(file['body_text'][:2], depth=3)
# texts = [(di['section'], di['text']) for di in file['body_text']]

# texts_di = {di['section']: "" for di in file['body_text']}

# for section, text in texts:

#     texts_di[section] += text



# pprint(list(texts_di.keys()))
# body = ""



# for section, text in texts_di.items():

#     body += section

#     body += "\n\n"

#     body += text

#     body += "\n\n"



# print(body[:3000])
# print(format_body(file['body_text'])[:3000])
# print(all_files[0]['metadata'].keys())
# print(all_files[0]['metadata']['title'])
# authors = all_files[0]['metadata']['authors']

# pprint(authors[:3])
# for author in authors:

#     print("Name:", format_name(author))

#     print("Affiliation:", format_affiliation(author['affiliation']))

#     print()
# pprint(all_files[4]['metadata'], depth=4)
# authors = all_files[4]['metadata']['authors']

# print("Formatting without affiliation:")

# print(format_authors(authors, with_affiliation=False))

# print("\nFormatting with affiliation:")

# print(format_authors(authors, with_affiliation=True))