import numpy as np 

import pandas as pd 

from pathlib import Path

import pandas as pd

import requests

from requests.exceptions import HTTPError, ConnectionError

from ipywidgets import interact

import ipywidgets as widgets

import nltk

from nltk.corpus import stopwords

# nltk.download("punkt")

pd.set_option("display.min_rows", 15)

pd.set_option("display.max_rows", 101)

pd.set_option("display.max_columns", 101)

pd.set_option('max_colwidth', 1000)

import os

import sys



# Where are all the files located

input_dir = '../input/CORD-19-research-challenge/2020-03-13'



# The all sources metadata file

src_metadata_file=os.path.join(input_dir, 'all_sources_metadata_2020-03-13.csv')

metadata = pd.read_csv(src_metadata_file, 

                      dtype={'Microsoft Academic Paper ID': str,

                             'pubmed_id': str})



# Convert the doi to a url

def doi_url(d): return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'

metadata.doi = metadata.doi.fillna('').apply(doi_url)

metadata.dtypes
metadata['retracted']=metadata.title.str.contains('retracted', case=False, na=False)

metadata[metadata['retracted']]
# records missing abstracts



missing_data=metadata[metadata.abstract.isna()] #['doi'].reset_index()

print(missing_data)
# Fill in missing cells using the doi address 



from bs4 import BeautifulSoup





def format_author(name):    # `Firstname Lastname` to `Lastname, Firstname;`

    fname,lname=name.split(' ',1)

    return '{}, {};'.format(lname,fname)



def fill_missing_medrxiv(row):

    url=row.doi

    print('Getting ' + url)

    r = requests.get(url,timeout=6)

    soup=BeautifulSoup(r.content, "html.parser")



    row.abstract=soup.find(id="p-2").text if soup.find(id="p-2") else ''

    authors=set([s.text for s in soup.find_all(class_="highwire-citation-author")])

    row.authors='\n'.join(list(map(format_author,authors)))

    row.title=soup.find(id="page-title").text if soup.find(id="page-title") else ''



    return row 



filled_medrxiv=missing_data[missing_data['source_x'].str.contains('medrxiv')]



filled_medrxiv=filled_medrxiv.apply(lambda x: fill_missing_medrxiv(x), axis=1)



# save extracted data

filled_medrxiv.to_csv('../output/kaggle/working/missing-medrxiv-metadata.tsv',sep='\t')

# print(filled_medrxiv)
tasks = [('What is known about transmission, incubation, and environmental stability?', 

        'transmission incubation environment coronavirus'),

        ('What do we know about COVID-19 risk factors?', 'risk factors'),

        ('What do we know about virus genetics, origin, and evolution?', 'genetics origin evolution'),

        ('What has been published about ethical and social science considerations','ethics ethical social'),

        ('What do we know about diagnostics and surveillance?','diagnose diagnostic surveillance'),

        ('What has been published about medical care?', 'medical care'),

        ('What do we know about vaccines and therapeutics?', 'vaccines vaccine vaccinate therapeutic therapeutics')] 

tasks = pd.DataFrame(tasks, columns=['Task', 'Keywords'])