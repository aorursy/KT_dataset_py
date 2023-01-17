import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path, PurePath

import pandas as pd

import requests

from requests.exceptions import HTTPError, ConnectionError

from ipywidgets import interact

import ipywidgets as widgets

import nltk

from nltk.corpus import stopwords

nltk.download("punkt")

pd.options.display.max_colwidth = 500
# Where are all the files located

input_dir = PurePath('../input/CORD-19-research-challenge/2020-03-13')



# The all sources metadata file

metadata = pd.read_csv(input_dir / 'all_sources_metadata_2020-03-13.csv', 

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



def show_metadata(ShowAllColumns=False):

    return metadata if ShowAllColumns else metadata[METADATA_COLS]



# Use ipywidgets to limit the sources

interact(show_metadata);
def get(url, timeout=6):

    try:

        r = requests.get(url, timeout=timeout)

        return r.text

    except ConnectionError:

        print(f'Cannot connect to {url}')

        print(f'Remember to turn Internet ON in the Kaggle notebook settings')

    except HTTPError:

        print('Got http error', r.status, r.text)



class ResearchPapers:

    

    def __init__(self, metadata: pd.DataFrame):

        self.metadata = metadata

        

    def __getitem__(self, item):

        return Paper(self.metadata.iloc[item])

    

    def __len__(self):

        return len(self.metadata)

    

    def head(self, n):

        return ResearchPapers(self.metadata.head(n).copy().reset_index(drop=True))

    

    def tail(self, n):

        return ResearchPapers(self.metadata.tail(n).copy().reset_index(drop=True))

    

    def abstracts(self):

        '''

        :return: a list of the abstracts

        '''

        return self.metadata.abstract.dropna()

    

    def titles(self):

        return self.metadata.title.dropna()

        

    def _repr_html_(self):

        return self.metadata._repr_html_()

    

class Paper:

    

    '''

    A single research paper

    '''

    def __init__(self, item):

        self.paper = item.to_frame().fillna('')

        self.paper.columns = ['Value']

    

    def doi(self):

        return self.paper.loc['doi'].values[0]

    

    def html(self):

        '''

        Load the paper from doi.org and display as HTML. Requires internet to be ON

        '''

        text = get(self.doi())

        return widgets.HTML(text)

    

    def text(self):

        '''

        Load the paper from doi.org and display as text. Requires Internet to be ON

        '''

        text = get(self.doi())

        return text

    

    def abstract(self):

        return self.paper.loc['abstract'].values[0]

    

    def title(self):

        return self.paper.loc['title'].values[0]

    

    def authors(self, split=False):

        '''

        Get a list of authors

        '''

        authors = self.paper.loc['authors'].values[0]

        if not authors:

            return []

        if not split:

            return authors

        if authors.startswith('['):

            authors = authors.lstrip('[').rstrip(']')

            return [a.strip().replace("\'", "") for a in authors.split("\',")]

        

        # Todo: Handle cases where author names are separated by ","

        return [a.strip() for a in authors.split(';')]

        

    def _repr_html_(self):

        return self.paper._repr_html_()

    



papers = ResearchPapers(metadata)

papers[1620].authors(split=True)
papers[0]
papers[0].html()
papers[0].text()[:1000]
papers.head(2)
papers.head(2).abstracts()
papers.head(2).titles()
english_stopwords = list(set(stopwords.words('english')))



def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words if word.isalnum() 

                                             and not word in english_stopwords

                                             and not (word.isnumeric() and len(word) < 4)]))

def preprocess(string):

    return tokenize(string.lower())



class SearchResults:

    

    def __init__(self, 

                 data: pd.DataFrame,

                 columns = None):

        self.results = data

        if columns:

            self.results = self.results[columns]

            

    def __getitem__(self, item):

        return Paper(self.results.loc[item])

        

    def _repr_html_(self):

        #display_cols = [col for col in self.results.columns if not col == 'index']

        return self.results._repr_html_()



class WordTokenIndex:

    

    def __init__(self, 

                 metadata: pd.DataFrame, 

                 columns=['title', 'abstract', 'doi', 'authors', 'journal', 'has_full_text']):

        self.metadata = metadata

        self.index = metadata.abstract.fillna('').apply(preprocess).to_frame()

        self.index.columns = ['terms']

        self.index.index = metadata.index

        self.columns = columns

    

    def search(self, search_string):

        search_terms = preprocess(search_string)

        result_index = self.index.terms.apply(lambda terms: any(i in terms for i in search_terms))

        results = self.metadata[result_index].copy().reset_index().rename(columns={'index':'paper'})

        return SearchResults(results, self.columns + ['paper'])

        
word_token_index = WordTokenIndex(metadata.head(10000))
results = word_token_index.search('Guidance')

results
results[3].title()
tasks = [('What is known about transmission, incubation, and environmental stability?', 

        'transmission incubation environment'),

        ('What do we know about COVID-19 risk factors?', 'risk factors'),

        ('What do we know about virus genetics, origin, and evolution?', 'genetics origin evolution'),

        ('What has been published about ethical and social science considerations','ethics ethical social'),

        ('What do we know about diagnostics and surveillance?','diagnose diagnostic surveillance'),

        ('What has been published about medical care?', 'medical care'),

        ('What do we know about vaccines and therapeutics?', 'vaccines vaccine vaccinate therapeutic therapeutics'),

        ('Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.', 'EUA, CLIA') ] 

tasks = pd.DataFrame(tasks, columns=['Task', 'Keywords'])
def show_task(Task):

    print(Task)

    keywords = tasks[tasks.Task == Task].Keywords.values[0]

    search_results = word_token_index.search(keywords)

    return search_results

    

results = interact(show_task, Task = tasks.Task.tolist());