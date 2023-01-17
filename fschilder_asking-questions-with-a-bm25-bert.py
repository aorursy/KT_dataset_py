!pip install rank_bm25 nltk
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path, PurePath

import pandas as pd

import requests

from requests.exceptions import HTTPError, ConnectionError

from ipywidgets import interact

import ipywidgets as widgets

from rank_bm25 import BM25Okapi

import nltk

from nltk.corpus import stopwords

nltk.download("punkt")

import re
from ipywidgets import interact

import ipywidgets as widgets

import pandas as pd



def set_column_width(ColumnWidth, MaxRows):

    pd.options.display.max_colwidth = ColumnWidth

    pd.options.display.max_rows = MaxRows

    print('Set pandas dataframe column width to', ColumnWidth, 'and max rows to', MaxRows)

    

interact(set_column_width, 

         ColumnWidth=widgets.IntSlider(min=50, max=400, step=50, value=200),

         MaxRows=widgets.IntSlider(min=50, max=500, step=100, value=100));
# Where are all the files located

input_dir = PurePath('../input/CORD-19-research-challenge')



list(Path(input_dir).glob('*'))
metadata_path = input_dir / 'metadata.csv'

metadata = pd.read_csv(metadata_path,

                               dtype={'Microsoft Academic Paper ID': str,

                                      'pubmed_id': str})



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



# Convert the doi to a url

def doi_url(d): 

    return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'





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

        if self.doi():

            url = doi_url(self.doi()) 

            text = get(url)

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
from rank_bm25 import BM25Okapi
english_stopwords = list(set(stopwords.words('english')))



def strip_characters(text):

    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)

    t = re.sub('/', ' ', t)

    t = t.replace("'",'')

    return t



def clean(text):

    t = text.lower()

    t = strip_characters(t)

    return t



def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words 

                     if len(word) > 1

                     and not word in english_stopwords

                     and not (word.isnumeric() and len(word) is not 4)

                     and (not word.isnumeric() or word.isalpha())] )

               )



def preprocess(text):

    t = clean(text)

    tokens = tokenize(t)

    return tokens



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

    

class RankBM25Index:

    

    def __init__(self, corpus: pd.DataFrame, columns=SEARCH_DISPLAY_COLUMNS):

        self.corpus = corpus

        self.columns = columns

        raw_search_str = self.corpus.abstract.fillna('') + ' ' + self.corpus.title.fillna('')

        self.index = raw_search_str.apply(preprocess).to_frame()

        self.index.columns = ['terms']

        self.index.index = self.corpus.index

        self.bm25 = BM25Okapi(self.index.terms.tolist())

        

    def search(self, search_string, n=4):

        search_terms = preprocess(search_string)

        doc_scores = self.bm25.get_scores(search_terms)

        ind = np.argsort(doc_scores)[::-1][:n]

        results = self.corpus.iloc[ind][self.columns]

        results['Score'] = doc_scores[ind]

        results = results[results.Score > 0]

        return SearchResults(results.reset_index(), self.columns + ['Score'])

    

bm25_index = RankBM25Index(metadata)
results = bm25_index.search('cruise ship')

results
bm25_index.search('sars-cov-2')
tasks = [('What is known about transmission, incubation, and environmental stability?', 

        'transmission incubation environment coronavirus'),

        ('What do we know about COVID-19 risk factors?', 'risk factors'),

        ('What do we know about virus genetics, origin, and evolution?', 'genetics origin evolution'),

        ('What has been published about ethical and social science considerations','ethics ethical social'),

        ('What do we know about diagnostics and surveillance?','diagnose diagnostic surveillance'),

        ('What has been published about medical care?', 'medical care'),

        ('What do we know about vaccines and therapeutics?', 'vaccines vaccine vaccinate therapeutic therapeutics')] 

tasks = pd.DataFrame(tasks, columns=['Task', 'Keywords'])
tasks
def show_task(Task):

    print(Task)

    keywords = tasks[tasks.Task == Task].Keywords.values[0]

    search_results = bm25_index.search(keywords, n=10)

    return search_results

    

results = interact(show_task, Task = tasks.Task.tolist());
from transformers import pipeline



nlp = pipeline("question-answering")

questions =  [('What is the mean incubation time for COVID-19?', 

        'mean incubation time covid-19'),

        ('How is COVID-19 transmitted?', 'transmission covid-19'),

        ('How stable is COVID-19 in different environments?', 'environment covid-19'),

        ] 

questions = pd.DataFrame(questions, columns=['Question', 'Keywords'])
def get_answer(answers):

    answers_sorted = sorted(answers, reverse= True,key = lambda x: (x['score']))

    #print("the answer is "+answers_sorted[0]['answer'])

    #print(answers_sorted[0]['abstract'])

    

    return pd.DataFrame(answers_sorted)



def show_answer(Question):

    print(Question)

    keywords = questions[questions.Question == Question].Keywords.values[0]

    search_results = bm25_index.search(keywords, n=10)

    answers = []

    for abstract in search_results.results['abstract']:

        #print(abstract)

        answer = nlp(question="What is the mean incubation time for COVID-19?", context=abstract)

        answer['abstract'] = abstract

        #print(answer)

        answers.append(answer)

    return get_answer(answers)

    

results = interact(show_answer, Question = questions.Question.tolist());
import torch

bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')

#bart.eval()
summarizer = pipeline("summarization", model=bart)

summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)