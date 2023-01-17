import os

import json

from pprint import pprint

from copy import deepcopy



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

import re
#Helper Functions by https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv



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

    clean_df = clean_df.drop(['raw_authors', 'raw_bibliography'], axis=1)

    clean_df.head()

    

    return clean_df
pmc_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/pmc_custom_license/pmc_custom_license/'

pmc_files = load_files(pmc_dir)

pmc_df = generate_clean_df(pmc_files)

print(pmc_df.shape)

pmc_df.head()
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'

biorxiv_files = load_files(biorxiv_dir)

biorxiv_df = generate_clean_df(biorxiv_files)

print(biorxiv_df.shape)

biorxiv_df.head()
noncomm_dir="/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/"

noncomm_files = load_files(noncomm_dir)

noncomm_df = generate_clean_df(noncomm_files)

print(noncomm_df.shape)

noncomm_df.head()
comm_dir="/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/"

comm_files = load_files(comm_dir)

comm_df = generate_clean_df(comm_files)

print(comm_df.shape)

comm_df.head()
#Function to find list of all titles with the occurence of provided word in the text

def wordcount(df,word):

    word_dict={}

    for i in range(len(df["text"])):

        if word in df["text"][i].lower():

            if df["title"][i]=="":

                word_dict[df["paper_id"][i]]=sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), df["text"][i],re.IGNORECASE))

            else:

                word_dict[df["title"][i]]=sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), df["text"][i],re.IGNORECASE))

            

    return {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1],reverse = True)}
biorxiv_dict=wordcount(biorxiv_df,"transmission")

print("Occurence of word 'transmission' in papers of biorxiv_medrxiv folder")

{k: biorxiv_dict[k] for k in list(biorxiv_dict)[:10]}
comm_dict=wordcount(comm_df,"transmission")

print("Occurence of word 'transmission' in papers of comm_use_subset folder")

{k: comm_dict[k] for k in list(comm_dict)[:10]}
comm_dict=wordcount(comm_df,"drug")

print("Occurence of word 'drug' in papers of comm_use_subset folder")

{k: comm_dict[k] for k in list(comm_dict)[:10]}
noncomm_dict=wordcount(noncomm_df,"drug")

print("Occurence of word 'drug' in papers of noncomm_use_subset folder")

{k: noncomm_dict[k] for k in list(noncomm_dict)[:10]}
comm_dict=wordcount(comm_df,"vaccine")

print("Occurence of word 'vaccine' in papers of comm_use_subset folder")

{k: comm_dict[k] for k in list(comm_dict)[:10]}
all_df=pmc_df.copy()

all_df=all_df.append(biorxiv_df)

all_df=all_df.append(noncomm_df)

all_df=all_df.append(comm_df)

print(all_df.shape)

all_df.head()
all_df.insert(7,'tags',"")

all_df.head()
possible_tags=['transmission','incubation','environmental stability','risks factors','infections','genetics','origin','evolution','geographic','livestock',

              'socioeconomic','ai','clinical trials','social science','vaccines','therapeutics','surveillance','diagnostics','pharmaceutical','non-pharmaceutical']
for i, row in tqdm(all_df.iterrows()):

    tag={}

    for i in possible_tags:

        tag[i]=0

    for word in possible_tags:

        if word in row["text"].lower():

            tag[word]=sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), row["text"],re.IGNORECASE))

    tag={k: v for k, v in sorted(tag.items(), key=lambda item: item[1],reverse = True)}

    tag={k: tag[k] for k in list(tag)[:5]}

    tags=""

    for i in tag.items():

        if i[1]!=0:

            tags+=i[0]+", "

    tags=tags[:-2]

    row['tags']=tags
all_df.head()
def find_papers_from_tags(df,tags):

    """

    if you want to find paper with multiple tags please send them in a comma seperated way

    e.g "infections,origin,vaccines"

    """    

    new_df=pd.DataFrame(columns=["paper_id","title","authors","affiliations","abstract","text","bibliography","tags"])

    tags=tags.split(",")

    for i, row in tqdm(df.iterrows()):

        for tag in tags:

            if tag in row['tags']:

                new_df=new_df.append(row)

                break

    return new_df
new_df=find_papers_from_tags(all_df,"origin")

print(new_df.shape)

new_df.head()
new_df=find_papers_from_tags(all_df,"infections,origin,vaccines")

print(new_df.shape)

new_df.head()
def find_papers_from_all_tags(df,tags):

    """

    if you want to find paper with multiple tags please send them in a comma seperated way

    e.g "infections,origin,vaccines"

    """    

    new_df=pd.DataFrame(columns=["paper_id","title","authors","affiliations","abstract","text","bibliography","tags"])

    tags=tags.split(",")

    for i, row in tqdm(df.iterrows()):

        found_all=True

        for tag in tags:

            if tag not in row['tags']:

                found_all=False

        if found_all:    

            new_df=new_df.append(row)

    return new_df
new_df=find_papers_from_all_tags(all_df,"origin")

print(new_df.shape)

new_df.head()
new_df=find_papers_from_all_tags(all_df,"infections,origin,vaccines")

print(new_df.shape)

new_df.head()