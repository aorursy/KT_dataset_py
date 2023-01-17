import numpy as np

import pandas as pd

import json

import os
from os import walk

for (dirpath, dirnames, filenames) in walk('/kaggle/input'):

    print('Directory path: ', dirpath, 'Folder name: ', dirnames)
with open('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/5d3612cd95331a2f0c43ef9d03ce583b5ff8c995.json', 'r') as f:

    test = json.load(f)

test.keys()

# test['paper_id']

# test['metadata']['title']

# test['metadata']['authors'][0]['first', 'middle', 'last', 'suffix', 'affiliation', 'email']

# test['metadata']['authors'][0]['affiliation']['laboratory', 'institution', 'location']

# test['metadata']['authors'][0]['affiliation']['location']['addrLine', 'postCode', 'region', 'settlement', 'country']

# test['abstract'][0]['text', 'cite_spans', 'ref_spans', 'section']

# test['body_text'][0]['text', 'cite_spans', 'ref_spans', 'section']

# test['bib_entries']['BIBREF0', 'BIBREF2', '...']['ref_id', 'title', 'authors', 'year', 'venue', 'volume', 'issn', 'pages', 'other_ids']

# test['ref_entries']['FIGREF0', 'FIGREF1', '...'][]'text', 'latex', 'type']

# test['back_matter']
def affiliation_parsing(x: dict) -> str:

    """Parse affiliation into string."""

    current = []

    for key in ['laboratory', 'institution']:

        if x['affiliation'].get(key):  # could also use try, except

            current.append(x['affiliation'][key])

        else:

            current.append('')

    for key in ['addrLine', 'settlement', 'region', 'country', 'postCode']:

        if x['affiliation'].get('location'):

            if x['affiliation']['location'].get(key):

                current.append(x['affiliation']['location'][key])

        else:

            current.append('')

    return ', '.join(current)





def cite_parsing(x: list, key: str) -> list:

    """Parse references into lists for delimiting."""

    cites = [i[key] if i else '' for i in x]

    output = []

    for i in cites:

        if i:

            output.append(','.join([j['ref_id'] if j['ref_id'] else '' for j in i]))

        else:

            output.append('')

    return '|'.join(output)





def extract_key(x: list, key:str) -> str:

    if x:

        return ['|'.join(i[key] if i[key] else '' for i in x)]

    return ''



extract_func = lambda x, func: ['|'.join(func(i) for i in x)]

format_authors = lambda x: f"{x['first']} {x['last']}"

format_full_authors = lambda x: f"{x['first']} {''.join(x['middle'])} {x['last']} {x['suffix']}"

format_abstract = lambda x: "{}\n {}".format(x['section'], x['text'])

all_keys = lambda x, key: '|'.join(i[key] for i in x.values())
for path in ['biorxiv_medrxiv', 'comm_use_subset', 'custom_license', 'noncomm_use_subset']:

    json_files = [file for file in os.listdir(f'/kaggle/input/CORD-19-research-challenge/{path}/{path}') if file.endswith('.json')]

    df_list = []



    for js in json_files:

        with open(os.path.join(f'/kaggle/input/CORD-19-research-challenge/{path}/{path}', js)) as json_file:

            paper = json.load(json_file)

            print(path, js)

        paper_df = pd.DataFrame({

            'paper_id': paper['paper_id'],

            'title': paper['metadata']['title'],

            'authors': extract_func(paper['metadata']['authors'], format_authors),

            'full_authors': extract_func(paper['metadata']['authors'], format_full_authors),

            'affiliations': extract_func(paper['metadata']['authors'], affiliation_parsing),

            'emails': extract_key(paper['metadata']['authors'], 'email'),

            'abstract': extract_func(paper['abstract'], format_abstract),

            'abstract_cite_spans': cite_parsing(paper['abstract'], 'cite_spans'),

            'abstract_ref_spans': cite_parsing(paper['abstract'], 'ref_spans'),

            'body': extract_func(paper['body_text'], format_abstract),

            'body_cite_spans': cite_parsing(paper['body_text'], 'cite_spans'),

            'body_ref_spans': cite_parsing(paper['body_text'], 'ref_spans'),

            'bib_titles': all_keys(paper['bib_entries'], 'title'),

            'ref_captions': all_keys(paper['ref_entries'], 'text'),

            'back_matter': extract_key(paper['back_matter'], 'text')

        })

        df_list.append(paper_df)

    temp_df = pd.concat(df_list)

    temp_df.to_csv(f'/kaggle/working/{path}.csv', index=False)
df_list = []

for path in ['biorxiv_medrxiv', 'comm_use_subset', 'custom_license', 'noncomm_use_subset']:

    temp_df = pd.read_csv(f'/kaggle/working/{path}.csv')

    temp_df['dataset'] = path

    df_list.append(temp_df)

    

aggregate_df = pd.concat(df_list)

aggregate_df.to_csv(f'/kaggle/working/all_datasets.csv', index=False)
aggregate_df  # view the aggregated data