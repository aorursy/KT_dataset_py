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
clean_df.iloc[606]['text'] = ''
import gensim
def read_corpus(df, column, tokens_only=False):

    for i, line in enumerate(df[column]):

        

        tokens = gensim.parsing.preprocess_string(line)

        if tokens_only:

            yield tokens

        else:

            # For training data, add tags

            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
train_corpus = list(read_corpus(clean_df, 'text'))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=5, epochs=30)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
list_of_bullets_to_retrieve = ["Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",

                            "Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",

                            "Seasonality of transmission.",

                            "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",

                            "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",

                            "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).",

                            "Natural history of the virus and shedding of it from an infected person",

                            "Implementation of diagnostics and products to improve clinical processes",

                            "Disease models, including animal models for infection, disease and transmission",

                            "Tools and studies to monitor phenotypic change and potential adaptation of the virus",

                            "Immune response and immunity",

                            "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

                            "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",

                            "Role of the environment in transmission"

                           ]
def get_doc_vector(doc):

    tokens = gensim.utils.simple_preprocess(doc)

    vector = model.infer_vector(tokens)

    return vector
get_doc_vector(list_of_bullets_to_retrieve[1])
clean_df['text_vector'] = clean_df['text'].apply(get_doc_vector)

clean_df['title_vector'] = clean_df['title'].apply(get_doc_vector)

clean_df['abstract_vector'] = clean_df['abstract'].apply(get_doc_vector)





array_of_bullets = [get_doc_vector(cat) for cat in list_of_bullets_to_retrieve]
from scipy.spatial import cKDTree

vectors = clean_df['title_vector'].values.tolist()

kdt = cKDTree(vectors)
distances, indices = kdt.query(array_of_bullets, k=3)
indices
for i, info in enumerate(list_of_bullets_to_retrieve):

    print(f"\n\nBULLET POINT = {info}\n")

    abstracts = clean_df.loc[indices[i], 'abstract']

    texts = clean_df.loc[indices[i], 'text']

    titles = clean_df.loc[indices[i], 'title']

    dist = distances[i]

    for l in range(len(dist)):

        print(f"Text index = {indices[i][l]} \n distance to bullet = {dist[l]} \n Title: {titles.iloc[l]} \n Abstract Extract: {abstracts.iloc[l][:100]}\n Text Extract : {texts.iloc[l][:100]}\n\n", "="*100)
