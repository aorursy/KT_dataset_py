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
import json

from pprint import pprint

from copy import deepcopy



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm
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
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'

filenames = os.listdir(biorxiv_dir)

print("Number of articles retrieved from biorxiv:", len(filenames))
all_files = []



for filename in filenames:

    filename = biorxiv_dir + filename

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
print(format_body(file['body_text'])[:3000])
for author in authors:

    print("Name:", format_name(author))

    print("Affiliation:", format_affiliation(author['affiliation']))

    print()
pprint(all_files[4]['metadata'], depth=4)
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
clean_df.to_csv('biorxiv_clean.csv', index=False)
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

    clean_df.head()

    

    return clean_df
pmc_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/pmc_custom_license/pmc_custom_license/'

pmc_files = load_files(pmc_dir)

pmc_df = generate_clean_df(pmc_files)

pmc_df.head()
comm_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/'

comm_files = load_files(comm_dir)

comm_df = generate_clean_df(comm_files)

comm_df.head()
comm_df.to_csv('clean_comm_use.csv', index=False)
noncomm_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/'

noncomm_files = load_files(noncomm_dir)

noncomm_df = generate_clean_df(noncomm_files)

noncomm_df.head()
noncomm_df.to_csv('clean_noncomm_use.csv', index=False)