import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os # operating system

import re # regular expression operations



# json library contains functions to handle

# json-formatted files in python

import json



# the pprint module prints Python data

# structures in a "pretty" way

from pprint import pprint



# a deepcopy constructs a new compound

# object (ie. list, dictionary) and then,

# recursively, inserts copies into it of 

# the objects found in the original

from copy import deepcopy



# tqdm is a progress bar library

from tqdm.notebook import tqdm
# print the directories given X levels

def walklevel(some_dir, level=1):

    some_dir = some_dir.rstrip(os.path.sep)

    dir_list=[some_dir]

    new_dir_list=[]

    for i in range(0,level):

        for d in dir_list:

            parent_dir = d

            child_dirs=[d + "/"  + c for c in os.listdir(d)]

            for child in child_dirs:

                if os.path.isdir(child):

                    new_dir_list.append(child)

            

        dir_list = new_dir_list

        new_dir_list = []

    return(dir_list)
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

    

    # merge all text for the same section

    for section, text in texts:

        texts_di[section] += text

    

    # put both section and text in the `body`

    # we seperate both text and sections with

    # "\n\n". You may need to account for this

    # when processing the text

    body = ""



    for section, text in texts_di.items():

        

        # remove text citations like [18]

        text_ptn = r'\[[0-9]{1,2}\]'

        text = re.sub(text_ptn,"",text)

        

        # remove figure citations

        fig_ptn = r'\(Fig.*\)'

        text = re.sub(fig_ptn,"",text)

        

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
input_dir="/kaggle/input/CORD-19-research-challenge"

metadata = pd.read_csv(input_dir+"/metadata.csv")

metadata.full_text_file.unique()
json_dirs = walklevel(input_dir,2) 

for jd in json_dirs:

    print(jd)
filenames=[]

for jd in json_dirs:

    pdir = os.path.basename(os.path.dirname(jd))

    sub_filenames = os.listdir(jd + "/pdf_json/")

    print("Number of articles retrieved from "+

          pdir+": "+str(len(sub_filenames)))

    sub_file_paths = [jd + "/pdf_json/" + f for f in sub_filenames]

    filenames.extend(sub_file_paths)
all_files = []



for filename in filenames:

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
fileIdx=10

for k in (all_files[fileIdx].keys()):

    print(k+":")

    pprint (all_files[fileIdx][k], depth=2)

    print("\n\n")
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
clean_df.to_csv('json_clean.csv', index=False)