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



    body = []

    for section, text in texts_di.items():

        body.append(text.replace(',', ''))

    

    return body



def format_key(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    key = []

    cnt = 0

    for section, text in texts_di.items():

        #tmp =''.join(filter(str.isdigit, section))

        #if not tmp: tmp = str(10001+cnt)

        key.append(cnt)

        cnt = cnt+1

    

    return key



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

            format_key(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 'key',

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

filenames = os.listdir(biorxiv_dir)

print("Number of articles retrieved from biorxiv:", len(filenames))
all_files = []



for filename in filenames:

    filename = biorxiv_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
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

        format_key(file['body_text']),

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

    'key',

    'bibliography',

    'raw_authors',

    'raw_bibliography'

]



clean_df = pd.DataFrame(cleaned_files, columns=col_names)

clean_df.head()
#clean_df.to_csv('/kaggle/working/biorxiv_clean.csv', index=False)
pmc_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'

pmc_files = load_files(pmc_dir)

pmc_df = generate_clean_df(pmc_files)

pmc_df.head()
#pmc_df.to_csv('/kaggle/working/clean_cust.csv', index=False)
comm_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'

comm_files = load_files(comm_dir)

comm_df = generate_clean_df(comm_files)

comm_df.head()
#comm_df.to_csv('/kaggle/working/clean_comm_use.csv', index=False)
noncomm_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'

noncomm_files = load_files(noncomm_dir)

noncomm_df = generate_clean_df(noncomm_files)

noncomm_df.head()
#noncomm_df.to_csv('/kaggle/working/clean_noncomm_use.csv', index=False)
print(noncomm_df.shape)
frames = [clean_df, pmc_df, comm_df, noncomm_df]

result = pd.concat(frames)

print(result.shape)
result.abstract[(result['abstract'].map(lambda d: len(d)) == 0)] = np.NaN

result.title[(result['title'].map(lambda d: len(d)) == 0)] = np.NaN

result.text[(result['text'].map(lambda d: len(d)) == 0)] = np.NaN
metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

print(f'meta data shape: {metadata.shape}')

has_title = metadata.title.apply(lambda x: str(x)!='nan')

metadata = metadata.iloc[has_title.values,:]

print(f'meta data shape after dropping docs without titles: {metadata.shape}')

print(metadata.head(10))
duplicate_paper = ~(metadata.title.isnull() | metadata.abstract.isnull()) & (metadata.duplicated(subset=['title', 'abstract']))

metadata = metadata[~duplicate_paper].reset_index(drop=True)
print(f'meta data shape after dropping dupes: {metadata.shape}')
metadata.isnull().sum()
temp_df = pd.DataFrame(columns=col_names)
temp_df['cord_uid'] = metadata['cord_uid']

temp_df['paper_id'] = metadata['sha']

temp_df['title'] = metadata['title']

temp_df['authors'] = metadata['authors']

temp_df['abstract'] = metadata['abstract']

temp_df['url'] = metadata['url']
result1 = result.drop(columns=['affiliations', 'bibliography','raw_authors','raw_bibliography'])

temp_df1 = temp_df.drop(columns=['affiliations', 'bibliography','raw_authors','raw_bibliography'])
temp = pd.concat([temp_df,result],sort=False)
temp1 = pd.merge(temp_df1, result1, left_on='title', right_on='title', how='outer')
temp1
temp2 = temp1.drop_duplicates(subset=['title']).reset_index(drop=True)

temp2.paper_id_x.fillna(temp2.paper_id_y, inplace=True)

temp2.authors_x.fillna(temp2.authors_y, inplace=True)

temp2.abstract_x.fillna(temp2.abstract_y, inplace=True)

temp2.text_x.fillna(temp2.text_y, inplace=True)

temp2.key_x.fillna(temp2.key_y, inplace=True)



df = temp2[temp2.columns.drop(list(temp2.filter(regex='_y')))]

df = df.rename(columns={"paper_id_x": "paper_id", "authors_x": "authors","abstract_x": "abstract", "text_x":"text","key_x":"key"})
df.columns
corpus = df.dropna(subset=['title','abstract','text'], how='all')
corpus
corpus.to_csv('/kaggle/working/combined_data3.csv', index=False)