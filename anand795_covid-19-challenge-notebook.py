from IPython.display import Image

Image("../input/ethics/ethics.png")
Image("../input/searchbar/searchbar.png")
!pip install rank_bm25 nltk
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
len(metadata)
# Some papers are duplicated since they were collected from separate sources. Thanks Joerg Rings

duplicate_paper = ~(metadata.title.isnull() | metadata.abstract.isnull()) & (metadata.duplicated(subset=['title', 'abstract']))

metadata = metadata[~duplicate_paper].reset_index(drop=True)
len(metadata)
def get(url, timeout=6):

    try:

        r = requests.get(url, timeout=timeout)

        return r.text

    except ConnectionError:

        print(f'Cannot connect to {url}')

        print(f'Remember to turn Internet ON in the Kaggle notebook settings')

    except HTTPError:

        print('Got http error', r.status, r.text)



class DataHolder:

    '''

    A wrapper for a dataframe with useful functions for notebooks

    '''

    def __init__(self, data: pd.DataFrame):

        self.data = data

        

    def __len__(self): return len(self.data)

    def __getitem__(self, item): return self.data.loc[item]

    def head(self, n:int): return DataHolder(self.data.head(n).copy())

    def tail(self, n:int): return DataHolder(self.data.tail(n).copy())

    def _repr_html_(self): return self.data._repr_html_()

    def __repr__(self): return self.data.__repr__()





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
from rank_bm25 import BM25Okapi
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

    

    def __len__(self):

        return len(self.results)

        

    def _repr_html_(self):

        return self.results._repr_html_()



SEARCH_DISPLAY_COLUMNS = ['title', 'abstract', 'doi', 'authors', 'journal']



class WordTokenIndex:

    

    def __init__(self, 

                 corpus: pd.DataFrame, 

                 columns=SEARCH_DISPLAY_COLUMNS):

        self.corpus = corpus

        raw_search_str = self.corpus.abstract.fillna('') + ' ' + self.corpus.title.fillna('')

        self.index = raw_search_str.apply(preprocess).to_frame()

        self.index.columns = ['terms']

        self.index.index = self.corpus.index

        self.columns = columns

    

    def search(self, search_string):

        search_terms = preprocess(search_string)

        result_index = self.index.terms.apply(lambda terms: any(i in terms for i in search_terms))

        results = self.corpus[result_index].copy().reset_index().rename(columns={'index':'paper'})

        return SearchResults(results, self.columns + ['paper'])
class RankBM25Index(WordTokenIndex):

    

    def __init__(self, corpus: pd.DataFrame, columns=SEARCH_DISPLAY_COLUMNS):

        super().__init__(corpus, columns)

        self.bm25 = BM25Okapi(self.index.terms.tolist())

        

    def search(self, search_string, n=4):

        search_terms = preprocess(search_string)

        doc_scores = self.bm25.get_scores(search_terms)

        ind = np.argsort(doc_scores)[::-1][:n]

        results = self.corpus.iloc[ind][self.columns]

        results['Score'] = doc_scores[ind]

        results = results[results.Score > 0]

        return SearchResults(results.reset_index(), self.columns + ['Score'])

    

bm25 = RankBM25Index(metadata.head(100))

bm25.search('cruise')
bm25_index = RankBM25Index(metadata.head(10000))
results = bm25_index.search('cruise ship')

results
results[3].title()
tasks = [('What is known about transmission, incubation, and environmental stability?', 

        'transmission incubation environment coronavirus'),

        ('What do we know about COVID-19 risk factors?', 'risk factors'),

        ('What do we know about virus genetics, origin, and evolution?', 'genetics origin evolution'),

        ('What has been published about ethical and social science considerations','ethics ethical social'),

        ('What do we know about diagnostics and surveillance?','diagnose diagnostic surveillance'),

        ('What has been published about medical care?', 'medical care'),

        ('What do we know about vaccines and therapeutics?', 'vaccines vaccine vaccinate therapeutic therapeutics')] 

tasks = pd.DataFrame(tasks, columns=['Task', 'Keywords'])
def show_task(Task):

    print(Task)

    keywords = tasks[tasks.Task == Task].Keywords.values[0]

    search_results = bm25_index.search(keywords, n=200)

    return search_results

    

results = interact(show_task, Task = tasks.Task.tolist());
from IPython.display import display



def search_papers(SearchTerms: str):

    search_results = bm25_index.search(SearchTerms, n=10)

    if len(search_results) > 0:

        display(search_results) 

    return search_results



searchbar = widgets.interactive(search_papers, SearchTerms='cruise ship')

searchbar
searchbar.result[0]
Image("../input/codeforcanada/code_for_canada.jpg")