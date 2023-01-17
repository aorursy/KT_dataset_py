WD = '/kaggle/input/CORD-19-research-challenge/2020-03-13/'



# Imports

import pandas as pd

from ipywidgets import interact

# !cat "{WD+'json_schema.txt'}"
!cat "{WD+'all_sources_metadata_2020-03-13.readme'}"
# The all sources metadata file

metadata = pd.read_csv(WD + "all_sources_metadata_2020-03-13.csv", 

                      dtype={'Microsoft Academic Paper ID': str,

                             'pubmed_id': str})



# Convert the doi to a url

def doi_url(d): return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'

metadata.doi = metadata.doi.fillna('').apply(doi_url)



# Set the abstract to the paper title if it is null

metadata.abstract = metadata.abstract.fillna(metadata.title)



# A list of columns to limit the display

METADATA_COLS = ['title', 'abstract', 'doi', 'publish_time',

                 'authors', 'journal', 'has_full_text']



def show_metadata(ShowAllColumns=False, show_head=True):

    meta_temp = metadata.head() if show_head == True else metadata

    return meta_temp if ShowAllColumns else meta_temp[METADATA_COLS]



# Use ipywidgets to limit the sources. 

interact(show_metadata);