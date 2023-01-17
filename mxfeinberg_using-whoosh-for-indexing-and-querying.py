import os

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
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'

filenames = os.listdir(biorxiv_dir)

print("Number of articles retrieved from biorxiv:", len(filenames))
all_files = []



for filename in filenames:

    filename = biorxiv_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
texts = [(di['section'], di['text']) for di in file['body_text']]

texts_di = {di['section']: "" for di in file['body_text']}

for section, text in texts:

    texts_di[section] += text
body = ""



for section, text in texts_di.items():

    body += section

    body += "\n\n"

    body += text

    body += "\n\n"
authors = all_files[0]['metadata']['authors']
authors = all_files[4]['metadata']['authors']

print("Formatting without affiliation:")

print(format_authors(authors, with_affiliation=False))

print("\nFormatting with affiliation:")

print(format_authors(authors, with_affiliation=True))
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
#clean_df.to_csv('biorxiv_clean.csv', index=False)
pmc_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/pmc_custom_license/pmc_custom_license/'

pmc_files = load_files(pmc_dir)

pmc_df = generate_clean_df(pmc_files)

pmc_df.head()
#pmc_df.to_csv('clean_pmc.csv', index=False)
comm_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/'

comm_files = load_files(comm_dir)

comm_df = generate_clean_df(comm_files)

comm_df.head()
#comm_df.to_csv('clean_comm_use.csv', index=False)
noncomm_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/'

noncomm_files = load_files(noncomm_dir)

noncomm_df = generate_clean_df(noncomm_files)

noncomm_df.head()
!pip install Whoosh
from whoosh.index import create_in

from whoosh.fields import *
schema = Schema(paper_id = TEXT(stored=True), title=TEXT(stored=True), abstract = TEXT(stored = True), content = TEXT(stored = True))
ix = create_in(".", schema)
writer = ix.writer()
papers = [clean_df, pmc_df, comm_df, noncomm_df]

#merged_papers = pd.concat(papers)
for paper_set in papers:

    for index, row in paper_set.iterrows():

        writer.add_document(paper_id = row['paper_id'],

                            title    = row['title'],

                            abstract = row['abstract'],

                            content  = row['text']

                           )

writer.commit()
from whoosh.qparser import QueryParser

from whoosh.query import *
searcher = ix.searcher()

proposed_time_strings = []

with ix.searcher() as searcher:

    parser = QueryParser("content", ix.schema)

    querystring = u"human incubation AND(content:period OR content:time) AND(content:corona) AND(content:virus)"# AND(content:months OR content:weeks OR content:days OR content:hours OR content:minutes OR content:seconds)"

    myquery = parser.parse(querystring)

    print(myquery)

    results = searcher.search(myquery)

    results.fragmenter.surround = 500

    results.fragmenter.maxchars = 1500

    print(len(results))

    for res in results:

        print(res['title'])

        #if(str.find(sub,start,end))

        res_strings = res.highlights("content", top = 5)

        print(res_strings)

        print("END__________________________________________________")

        #if(res_strings.find("days") or res_strings.find("minutes") or res_strings.find("hours") or res_strings.find("seconds") or res_strings.find("weeks")):

        proposed_time_strings.append(res_strings)

        #proposed_times.append([int(s) for s in res_strings.split() if s.isdigit()])

    

    

        