# installing the libraries needed (in my case they were already installed)



!pip install rank_bm25 nltk
# importing libraries



import numpy as np

import pandas as pd

from pathlib import Path, PurePath

import pandas as pd

import requests

from requests.exceptions import HTTPError, ConnectionError

from ipywidgets import interact

import ipywidgets as widgets

from IPython.display import display

from rank_bm25 import BM25Okapi

import nltk

from nltk.corpus import stopwords

nltk.download("punkt")

nltk.download("stopwords")

import re

import plotly.express as px
# adjusting the size of columns and rows

# I left the rows to 29500 because I want all the results to be shown



def set_column_width(ColumnWidth, MaxRows):

    pd.options.display.max_colwidth = ColumnWidth

    pd.options.display.max_rows = MaxRows

    print('Set pandas dataframe column width to', ColumnWidth, 'and max rows to', MaxRows)

    

interact(set_column_width, 

         ColumnWidth=widgets.IntSlider(min=50, max=400, step=50, value=200),

         MaxRows=widgets.IntSlider(min=50, max=29500, step=100, value=29500));
# Where are all the files located



input_dir = PurePath('../input/CORD-19-research-challenge')



list(Path(input_dir).glob('*'))
# import metadata



metadata_path = input_dir / 'metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str

})

meta_df.head()
# Convert the doi to a url



def doi_url(d): return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'

meta_df.doi = meta_df.doi.fillna('').apply(doi_url)
# Set the abstract to the paper title if it is null



meta_df.abstract = meta_df.abstract.fillna(meta_df.title)
# checking how many entries there are in the DataFrame



len(meta_df)
# Removing duplicated papers



duplicate_paper = ~(meta_df.title.isnull() | meta_df.abstract.isnull()) & (meta_df.duplicated(subset=['title', 'abstract']))

meta_df = meta_df[~duplicate_paper].reset_index(drop=True)
# checking how many entries there are in the DataFrame



len(meta_df)
# Creating a function to get the requests from an url



def get(url, timeout=6):

    try:

        r = requests.get(url, timeout=timeout)

        return r.text

    except ConnectionError:

        print(f'Cannot connect to {url}')

        print(f'Remember to turn Internet ON in the Kaggle notebook settings')

    except HTTPError:

        print('Got http error', r.status, r.text)
# Creating a wrapper for a DataFrame with useful functions for notebooks



class DataHolder:    

    def __init__(self, data: pd.DataFrame):

        self.data = data        

        

    def __len__(self): return len(self.data)

    

    def __getitem__(self, item): return self.data.loc[item]

    

    def head(self, n:int): return DataHolder(self.data.head(n).copy())

    

    def tail(self, n:int): return DataHolder(self.data.tail(n).copy())

    

    def _repr_html_(self): return self.data._repr_html_()

    

    def __repr__(self): return self.data.__repr__()
# Creating a wrapper for the entire dataset and provides useful functions to navigate through it



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
# Creating a wrapper for each research paper



class Paper:

    

    # single research paper 

    

    def __init__(self, item):

        self.paper = item.to_frame().fillna('')

        self.paper.columns = ['Value']

    

    def doi(self):

        return self.paper.loc['doi'].values[0]

    

    def html(self):

        

        # loads the paper from doi.org and displays it as HTML. Needs internet on

        

        text = get(self.doi())

        return widgets.HTML(text)

    

    def text(self):

        

        # loads the paper from doi.org and display as text. Needs internet on

        

        text = get(self.doi())

        return text

    

    def abstract(self):

        return self.paper.loc['abstract'].values[0]

    

    def title(self):

        return self.paper.loc['title'].values[0]

    

    def authors(self, split=False):

        

        # gets a list of authors

        

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

    



papers = ResearchPapers(meta_df)
abstracts = papers.head(29500).abstracts()

type(abstracts)
# Creating a list of stopwords in english



english_stopwords = list(set(stopwords.words('english')))
# Creating a function that cleans text of special characters



def strip_characters(text):

    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)

    t = re.sub('/', ' ', t)

    t = t.replace("'",'')

    return t
# Creating a function that makes text lowercase and uses the function created above



def clean(text):

    t = text.lower()

    t = strip_characters(t)

    return t
# Tokenize into individual tokens - words mostly



def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words 

                     if len(word) > 1

                     and not word in english_stopwords

                     and not (word.isnumeric() and len(word) is not 4)

                     and (not word.isnumeric() or word.isalpha())] )

               )
# Creating a function that cleans and tokenize texts



def preprocess(text):

    t = clean(text)

    tokens = tokenize(t)

    return tokens
# Creating a wrapper for the search results



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
# Defining column names of the search display



SEARCH_DISPLAY_COLUMNS = ['title', 'abstract', 'doi', 'authors', 'journal']
# Creating a wrapper for the word tokens which will be searched



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
# Creating the search index class



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
# Creating the search index



bm25_index = RankBM25Index(meta_df)
# Creating a list for the tasks in the challenge and it's keywords



tasks = [('What is known about transmission, incubation, and environmental stability?', 'transmission periods asymptomatic shedding persistance stability substrate surfaces diagnostics disease models tools immune immunity effectiveness strategy movement health care community PPE seasonality incubation environmental stability coronavirus'),

         ('What do we know about COVID-19 risk factors?', 'COVID-19 risk factors coronavirus smoking pre-existing pulmonary disease neonates pregnant women severity transmission dynamics susceptibility of populations public health mitigation measures that could be effective for control'),

         ('What do we know about virus genetics, origin, and evolution?', 'virus genetics origin evolution coronavirus real-time tracking genome dissemination diagnostics geographic temporal genomic strain nagoya protocol livestock field surveillance genetic sequencing receptor binding farmers wildlife SARS-CoV-2 animal host socioeconomic behavioral reduction risk'),

         ('Sample task with sample submission', 'sample submission geographic variation variations mutation evidence'),

         ('What do we know about vaccines and therapeutics?', 'drugs clinical bench trials less common viral inhabitors naproxen clarithromycin minocyclinethat Antibody-Dependent Enhancement (ADE) in vaccine recipients animal model predictive value therapeutic alternative model prioritize distribution expanding production universal vaccine standardize prophylaxis clinical studies enhanced disease immune response'),

         ('What do we know about non-pharmaceutical interventions?', 'guidance scale up NPI funding infrastructure authorities support authoritative collaboration health care delivery system capacity respond increase case design execution experiment DHS center excelence assessment school closure travel ban mass gathering social distancing spread communities barriers compliance intervention cost benefit race income disability age geographic location immigration status housing employment health insurance compliance underserved advice pandemicpolicy programmatic mitigate government service food distribuition supplies food household diagnose treatment'),

         ('What has been published about ethical and social science considerations?', 'effort articulate translate existing ethical principle standard salient issues COVID-19 thematic novel duplicate oversight sustained education access capacity WHO multidisciplinary research operational platform global network social science qualitative assessment framework local barrier enabler adherence public health measure prevention control surgical mask SRH school closure outbreak physical physiological underlying driver fear anxiety stigma misinformation rumor social media'),

         ('What do we know about diagnostics and surveillance?', 'widespread exposure immediate policy recommentation mitigation measure denominator testing mechanism sharing information demographics sampling asymptomatic disease serosurvey convalescent screening neutralizing antibodies ELISA diagnostic surveillance recruitment legal ethical communication public health official national guidance guidelines best practices state universities private laboratories tradeoff speed accuracy accessibility PCR specific entity assay private sector evolution genetic drift mutation reagent latency pathogen cytokines progression therapeutical policies roadmap coalition epidemic preparedness inovation CRISPR genomics rapid sequencing bioinformatic wildlife domestic risk'),

         ('What has been published about medical care?', 'nursing long term care medical staff shortage overwhelmed communities age-ajusted mortality Acute Respiratory Distress Syndrome ARDS organ failure viral etiology Extracorporeal membrane oxygenation ECMO mechanical ventilation frequency manifestation cardiomyopathy cardiac arrest regulatory standard EUA CLIA crisis care level elastomeric respirator N95 mask telemedicine barriers facilitators specific action state boundaries guidance oral medication AI real-time health care delivery valuate interventions risk factors outcomes best practices critical challenges innovative solutions technologies flow organization workforce protection allocation community-based resources payment supply chain natural history interventions steroids high flow oxygen'),

         ('What has been published about information sharing and inter-sectoral collaboration?', 'data-gathering standardized nomenclature sharing response information with planners providers mitigating barriers of information-sharing recruit support coordinate local expertise capacity public health emergency response integration federal state local public health surveillance systems investment baseline infrastructure preparedness high-risk population elderly health care workers guidelines easy understand risk disease population group misunderstanding containment mitigation action plan surveillance treatment marginalized disadvantaged data system incarcareted COVID-19 information prevention diagnosis coverage policies')]
# Transforming the list into a DataFrame



tasks = pd.DataFrame(tasks, columns=['Task', 'Keywords'])
# Creating a dropdown menu for each task



def show_task(Task):

    print(Task)

    keywords = tasks[tasks.Task == Task].Keywords.values[0]

    search_results = bm25_index.search(keywords, n=1000)

    display(search_results)

    return search_results

    

results = interact(show_task, Task = tasks.Task.tolist());
# Creating autocomplete search bar



def search_papers(SearchTerms: str):

    search_results = bm25_index.search(SearchTerms, n=1000)

    if len(search_results) > 0:

        display(search_results) 

    return search_results
# Autocomplete search bar



searchbar = widgets.interactive(search_papers, SearchTerms='ethic')

searchbar